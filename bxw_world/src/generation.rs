use crate::ecs::CLoadAnchor;
use crate::stdgen::StdGenerator;
use crate::worldmgr::*;
use crate::*;
use bxw_util::itertools::Itertools;
use bxw_util::taskpool::Task;
use std::any::Any;
use std::sync::Arc;
use std::time::Instant;

pub struct WorldBlocks {
    pub voxel_registry: Arc<VoxelRegistry>,
    pub generator: Arc<StdGenerator>,
    status_array: Vec<ChunkDataState>,
    compressed_storage: Vec<Option<Arc<VChunk>>>,
    cache: RefCell<VCache>,
}

impl WorldBlocks {
    pub fn new(voxel_registry: Arc<VoxelRegistry>, seed: u64) -> Self {
        Self {
            voxel_registry,
            generator: Arc::new(StdGenerator::new(seed)),
            status_array: Vec::new(),
            compressed_storage: Vec::new(),
            cache: Default::default(),
        }
    }

    pub fn get_vcache(&self) -> RefMut<'_, VCache> {
        self.cache.borrow_mut()
    }

    pub fn get_chunk(&self, world: &World, cpos: ChunkPosition) -> Option<Arc<VChunk>> {
        let idx = world.get_chunk_index(cpos);
        idx.and_then(|i| Some(self.compressed_storage[i].as_ref()?.clone()))
    }

    pub(crate) fn modify_chunk<'a, I: IntoIterator<Item = &'a VoxelChange>>(
        &mut self,
        world: &World,
        cpos: ChunkPosition,
        changes: I,
    ) {
        let mut cache = self.cache.borrow_mut();
        let cidx = world
            .get_chunk_index(cpos)
            .expect("Trying to modify a chunk that is not loaded");
        let chunk = cache.get_uncompressed_chunk_mut(world, self, cpos);
        if let Some(chunk) = chunk {
            for change in changes {
                let bidx = blockidx_from_blockpos(change.bpos);
                if chunk.blocks_yzx[bidx] == change.from {
                    chunk.blocks_yzx[bidx] = change.to;
                }
            }
            let mut vchunk = VChunk::new();
            vchunk.position = cpos;
            vchunk.compress(chunk);
            self.compressed_storage[cidx] = Some(Arc::new(vchunk));
        } else {
            panic!("Trying to modify a chunk that is not loaded");
        }
    }
}

impl ChunkDataHandler for WorldBlocks {
    fn status_array(&self) -> &Vec<ChunkDataState> {
        &self.status_array
    }

    fn status_array_mut(&mut self) -> &mut Vec<ChunkDataState> {
        &mut self.status_array
    }

    fn get_dependency(&self) -> Option<(usize, bool)> {
        None
    }

    fn get_data(&self, _world: &World, index: usize) -> AnyChunkData {
        self.compressed_storage[index]
            .clone()
            .map(|x| x as AnyChunkDataArc)
    }

    fn swap_data(&mut self, _world: &World, index: usize, new_data: AnyChunkData) -> AnyChunkData {
        let new_data = new_data.map(|d| d.downcast::<VChunk>().unwrap());
        let old_data = std::mem::replace(&mut self.compressed_storage[index], new_data);
        old_data.map(|x| x as AnyChunkDataArc)
    }

    fn resize_data(&mut self, _world: &World, new_size: usize) {
        self.compressed_storage.resize(new_size, None);
    }

    fn create_chunk_update_task(
        &mut self,
        world: &World,
        cpos: ChunkPosition,
        index: usize,
    ) -> Option<Task> {
        self.status_array[index] = ChunkDataState::Loading;
        let registry = self.voxel_registry.clone();
        let submit_channel = world.get_sync_task_channel();
        let worldgen = Arc::downgrade(&self.generator);
        Some(Task::new(
            move || {
                let _p_zone = bxw_util::tracy_client::Span::new(
                    "Chunk generate task",
                    "mainloop",
                    file!(),
                    line!(),
                    4,
                );
                let worldgen = match worldgen.upgrade() {
                    Some(g) => g,
                    None => return,
                };
                let pregen = Instant::now();
                let mut chunk = VChunk::new();
                chunk.position = cpos;
                let mut ucchunk = UncompressedChunk::new();
                ucchunk.position = cpos;
                worldgen.generate_chunk(&mut ucchunk, &registry);
                chunk.compress(&ucchunk);
                let chunk = Arc::new(chunk);
                drop(ucchunk);
                let postgen = Instant::now();
                let gentime = postgen.saturating_duration_since(pregen);
                bxw_util::debug_data::DEBUG_DATA
                    .wgen_times
                    .push_ns(gentime.as_nanos() as i64);
                submit_channel
                    .send(Box::new(move |world| {
                        let index = match world.get_chunk_index(cpos) {
                            Some(i) => i,
                            None => return,
                        };
                        let mut blocks = world.get_handler(CHUNK_BLOCK_DATA).borrow_mut();
                        let blocks: &mut Self = blocks.as_any_mut().downcast_mut().unwrap();
                        // request was cancelled
                        if blocks.status_array[index] == ChunkDataState::Unloaded {
                            return;
                        }
                        blocks.status_array[index] = ChunkDataState::Loaded;
                        blocks.compressed_storage[index] = Some(chunk);
                        blocks.cache.borrow_mut().uncompressed_chunks.pop(&cpos);
                    }))
                    .unwrap_or(());
            },
            false,
            false,
        ))
    }

    fn needs_loading_for_anchor(&self, _anchor: &CLoadAnchor) -> bool {
        true
    }

    fn serializable(&self) -> bool {
        true
    }

    fn serialize_data(&self, _world: &World, index: usize) -> Option<Vec<u8>> {
        let data = self.compressed_storage.get(index)?.as_ref()?;
        let VChunkData::QuickCompressed { vox } = &data.data;
        let mut out: Vec<u8> = Vec::with_capacity(vox.len() * 4);
        vox.iter()
            .map(|word| word.to_le_bytes())
            .for_each(|bytes| out.extend_from_slice(&bytes));
        Some(out)
    }

    fn deserialize_data(
        &mut self,
        world: &World,
        index: usize,
        data: &[u8],
    ) -> Result<AnyChunkData, &'static str> {
        if (data.len() % 4) != 0 {
            return Err("Invalid serialized data length");
        }
        let mut vox: Vec<u32> = Vec::with_capacity(data.len() / 4);
        vox.extend(
            data.iter()
                .tuples()
                .map(|(&a, &b, &c, &d)| u32::from_le_bytes([a, b, c, d])),
        );
        let new_data = VChunk {
            data: VChunkData::QuickCompressed { vox },
            position: world
                .get_chunk_position(index)
                .ok_or("Trying to deserialize a chunk without an assigned position")?,
        };
        let old_data = std::mem::replace(
            &mut self.compressed_storage[index],
            Some(Arc::new(new_data)),
        );
        Ok(old_data.map(|x| x as AnyChunkDataArc))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct VCache {
    uncompressed_chunks: LruCache<ChunkPosition, Box<UncompressedChunk>>,
}

impl Default for VCache {
    fn default() -> Self {
        Self {
            uncompressed_chunks: LruCache::new(64),
        }
    }
}

impl VCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn ensure_newest_cached(
        &mut self,
        world: &World,
        voxels: &WorldBlocks,
        cpos: ChunkPosition,
    ) -> Option<()> {
        let newest = voxels.get_chunk(world, cpos)?;
        let cached = self.uncompressed_chunks.get_mut(&cpos);
        if cached.is_some() {
            Some(())
        } else {
            self.uncompressed_chunks.put(cpos, newest.decompress());
            Some(())
        }
    }

    /// Cached
    pub fn get_uncompressed_chunk(
        &mut self,
        world: &World,
        voxels: &WorldBlocks,
        cpos: ChunkPosition,
    ) -> Option<&UncompressedChunk> {
        self.ensure_newest_cached(world, voxels, cpos)?;
        self.uncompressed_chunks
            .get(&cpos)
            .map(|x| x as &UncompressedChunk)
    }

    pub fn get_uncompressed_chunk_mut(
        &mut self,
        world: &World,
        voxels: &WorldBlocks,
        cpos: ChunkPosition,
    ) -> Option<&mut UncompressedChunk> {
        self.ensure_newest_cached(world, voxels, cpos)?;
        self.uncompressed_chunks
            .get_mut(&cpos)
            .map(|x| x as &mut UncompressedChunk)
    }

    pub fn peek_uncompressed_chunk(&self, cpos: ChunkPosition) -> Option<&UncompressedChunk> {
        self.uncompressed_chunks
            .peek(&cpos)
            .map(|x| x as &UncompressedChunk)
    }

    pub fn get_block(
        &mut self,
        world: &World,
        voxels: &WorldBlocks,
        bpos: BlockPosition,
    ) -> Option<VoxelDatum> {
        let cpos = chunkpos_from_blockpos(bpos);
        self.ensure_newest_cached(world, voxels, cpos)?;
        Some(self.uncompressed_chunks.get(&cpos)?.blocks_yzx[blockidx_from_blockpos(bpos)])
    }

    pub fn peek_block(&self, bpos: BlockPosition) -> Option<VoxelDatum> {
        let cpos = chunkpos_from_blockpos(bpos);
        Some(self.uncompressed_chunks.peek(&cpos)?.blocks_yzx[blockidx_from_blockpos(bpos)])
    }
}
