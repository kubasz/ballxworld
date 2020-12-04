#![allow(unused_variables)]
#![deny(unused_must_use)]

use crate::client::input::InputManager;
use crate::client::render::resources::RenderingResources;
use crate::client::render::ui::z::*;
use crate::client::render::ui::{
    gv2, GuiCmd, GuiControlStyle, GuiCoord, GuiOrderedCmd, GuiRect, GuiRenderer, GuiVec2,
    GUI_BLACK, GUI_WHITE,
};
use crate::client::render::{RenderingContext, VoxelRenderer};
use crate::client::world::{CameraSettings, ClientWorld};
use crate::config::Config;
use bxw_util::debug_data::DEBUG_DATA;
use bxw_util::math::*;
use bxw_util::*;
use bxw_world::blocks::register_standard_blocks;
use bxw_world::ecs::*;
use bxw_world::entities::player::PLAYER_EYE_HEIGHT;
use bxw_world::generation::WorldBlocks;
use bxw_world::BlockPosition;
use std::borrow::Cow;
use std::f64::consts::PI;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::client::render::voxrender::MeshDataHandler;
use bxw_util::change::Change;
use bxw_util::collider::AABB;
use bxw_util::direction::OctahedralOrientation;
use bxw_world::blocks::stdshapes::StdMeta;
use bxw_world::physics::SMALL_V_CUTOFF;
use bxw_world::physics::TIMESTEP as PHYSICS_FRAME_TIME;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug, Clone, Default)]
struct InputState {
    /// X (-Left,Right+) Y (-Down,Up+)
    walk: (f32, f32),
    look: (f32, f32),
    capture_mouse: bool,
    do_normal: bool,
}

pub fn client_main() {
    let sdl_ctx = sdl2::init().unwrap();
    let sdl_vid = sdl_ctx.video().unwrap();
    let mut sdl_timer = sdl_ctx.timer().unwrap();

    let mut cfg = Config::standard_load();
    if std::env::args().any(|a| a == "-renderdoc") {
        cfg.dbg_renderdoc = true;
        cfg.vk_debug_layers = false;
        log::warn!("Adjusting settings for renderdoc");
    }

    let task_pool = bxw_util::taskpool::TaskPool::new(cfg.performance_threads as usize);
    let mut rctx = Box::new(RenderingContext::new(&sdl_vid, &cfg));
    let rres = Arc::new(RenderingResources::load(&cfg, &mut rctx));
    let vctx = Rc::new(RefCell::new(VoxelRenderer::new(
        &cfg,
        &mut rctx,
        rres.clone(),
    )));
    let mut guictx = Box::new(GuiRenderer::new(&cfg, &mut rctx, rres.clone()));

    let mut vxreg: Box<bxw_world::voxregistry::VoxelRegistry> = Box::default();
    {
        let vctx = vctx.borrow();
        register_standard_blocks(&mut vxreg, &|nm| vctx.get_texture_id(nm));
    }
    let vxreg: Arc<bxw_world::voxregistry::VoxelRegistry> = Arc::from(vxreg);
    let (mut world, mut client_world) = ClientWorld::new_world("world".to_owned(), vxreg.clone());
    {
        let lp = client_world.local_player;
        let ents = world.ecs();
        let anchor: &CLoadAnchor = ents.get_component(lp).unwrap();
        let mut new_anchor = anchor.clone();
        new_anchor.radius = cfg.performance_load_distance;
        let change = [EntityChange {
            kind: EntityChangeKind::UpdateEntity(anchor.entity_id()),
            load_anchor: Change::Update {
                old: anchor.clone(),
                new: new_anchor,
            },
            ..Default::default()
        }];
        world.apply_entity_changes(&change);
    }
    world.replace_handler(
        bxw_world::worldmgr::CHUNK_MESH_DATA,
        Box::new(MeshDataHandler::new(vctx.clone(), &rctx.handles)),
    );
    let wgen = WorldBlocks::new(vxreg.clone(), 0);

    let pf_mult = 1.0 / sdl_timer.performance_frequency() as f64;
    let mut previous_frame_time = sdl_timer.performance_counter() as f64 * pf_mult;
    let mut physics_accum_time = 0.0f64;
    //

    let mut input_mgr = InputManager::new(&sdl_ctx);
    input_mgr.input_state.capture_input_requested = true;

    let mut look_pos: BlockPosition = zero();
    let mut look_precise_pos: Vector3<f64> = zero();
    let mut click_pos: Option<BlockPosition> = None;
    let mut click_datum: bxw_world::VoxelDatum = Default::default();
    let mut click_place = false;

    let mut event_pump = sdl_ctx.event_pump().unwrap();

    let i_placeable = [
        vxreg.get_definition_from_name("core:grass").unwrap(),
        vxreg.get_definition_from_name("core:snow_grass").unwrap(),
        vxreg.get_definition_from_name("core:dirt").unwrap(),
        vxreg.get_definition_from_name("core:stone").unwrap(),
        vxreg.get_definition_from_name("core:diamond_ore").unwrap(),
        vxreg.get_definition_from_name("core:debug").unwrap(),
        vxreg.get_definition_from_name("core:table").unwrap(),
    ];
    let mut i_place = 4;
    let mut i_orientation = 0;
    let i_destroy = vxreg.get_definition_from_name("core:void").unwrap();

    'running: loop {
        let current_frame_time = sdl_timer.performance_counter() as f64 * pf_mult;

        // avoid any nasty divisions by 0
        let frame_delta_time = 1.0e-6f64.max(current_frame_time - previous_frame_time);
        DEBUG_DATA.frame_times.push_sec(frame_delta_time);
        DEBUG_DATA
            .fps
            .store((1.0 / frame_delta_time) as u32, Ordering::Release);

        physics_accum_time += frame_delta_time;
        previous_frame_time = current_frame_time;
        let physics_frames = (physics_accum_time / PHYSICS_FRAME_TIME) as i32;
        {
            let (dyaw, dpitch) = (
                input_mgr.input_state.look.x as f64,
                input_mgr.input_state.look.y as f64,
            );
            input_mgr.input_state.look = zero();
            let CameraSettings::FPS { pitch, yaw } = &mut client_world.camera_settings;
            *pitch -= dpitch / 60.0;
            *pitch = f64::min(f64::max(*pitch, -PI / 4.0 + 0.01), PI / 4.0 - 0.01);
            *yaw -= dyaw / 60.0;
            *yaw %= 2.0 * PI;
            let (pitch, yaw) = (*pitch, *yaw);
        }
        if physics_frames > 0 {
            physics_accum_time -= f64::from(physics_frames) * PHYSICS_FRAME_TIME;
            let physics_frames = if physics_frames > 10 {
                log::warn!(
                    "Physics lagging behind, skipping {} ticks",
                    physics_frames - 1
                );
                1
            } else {
                physics_frames
            };
            let local_player = client_world.local_player;

            if let Some(bpos) = click_pos {
                click_pos = None;
                // place block
                let i_used = if click_place {
                    i_placeable[i_place]
                } else {
                    i_destroy
                };
                let shape = if input_mgr
                    .pressed_keys
                    .contains(&sdl2::keyboard::Keycode::Num1)
                {
                    1
                } else if input_mgr
                    .pressed_keys
                    .contains(&sdl2::keyboard::Keycode::Num2)
                {
                    2
                } else if input_mgr
                    .pressed_keys
                    .contains(&sdl2::keyboard::Keycode::Num3)
                {
                    3
                } else {
                    0
                };
                let meta = StdMeta::from_parts(shape, i_orientation).unwrap().to_meta();
                let change = bxw_world::worldmgr::VoxelChange {
                    bpos,
                    from: click_datum,
                    to: bxw_world::VoxelDatum::new(i_used.id(), meta),
                };
                world.apply_voxel_changes(&[change]);
            }

            for _pfrm in 0..physics_frames {
                // do physics tick
                let entities = world.ecs();
                let lp_loc: &CLocation = entities.get_component(local_player).unwrap();
                let mut new_loc: CLocation = lp_loc.clone();
                // position
                let &CameraSettings::FPS { pitch, yaw } = &client_world.camera_settings;
                let qyaw = Quaternion::from_polar_decomposition(1.0, yaw, Vector3::y_axis());
                let qpitch = Quaternion::from_polar_decomposition(1.0, pitch, Vector3::x_axis());
                new_loc.orientation = UnitQuaternion::new_normalize(qpitch * qyaw);

                let mview = glm::quat_to_mat3(&new_loc.orientation).transpose();

                let mut wvel = Vector3::new(
                    input_mgr.input_state.walk.x as f64,
                    0.0,
                    input_mgr.input_state.walk.y as f64,
                );
                wvel *= 6.0; // Walk 6m/s
                if input_mgr.input_state.noclip {
                    wvel *= 10.0;
                }
                if input_mgr.input_state.sprint.is_active() {
                    wvel *= 3.0;
                }
                let mut tvel = mview * wvel;
                let tspeed = tvel.magnitude();

                if input_mgr.input_state.noclip {
                    new_loc.position += tvel * PHYSICS_FRAME_TIME;
                    new_loc.velocity = tvel;
                } else {
                    tvel.y = 0.0;
                }
                let lp_phys: &CPhysics = entities.get_component(local_player).unwrap();
                let mut new_phys: CPhysics = lp_phys.clone();
                new_phys.control_target_velocity = if tspeed < SMALL_V_CUTOFF {
                    zero()
                } else {
                    tvel.normalize() * tspeed
                };
                new_phys.frozen = input_mgr.input_state.noclip;
                if input_mgr.input_state.jump.is_active() && new_phys.against_wall[2] {
                    new_phys.control_frame_impulse.y = 300.0;
                }
                let change = [EntityChange {
                    kind: EntityChangeKind::UpdateEntity(lp_loc.entity_id()),
                    location: Change::Update {
                        old: lp_loc.clone(),
                        new: new_loc,
                    },
                    physics: Change::Update {
                        old: lp_phys.clone(),
                        new: new_phys,
                    },
                    ..Default::default()
                }];
                world.apply_entity_changes(&change);
                bxw_world::physics::world_physics_tick(&mut world);
            }
        }

        world.main_loop_tick(&task_pool);
        task_pool.main_thread_tick();

        if let Some(mut fc) = rctx.frame_begin_prepass(&cfg, &client_world, frame_delta_time) {
            let mut vctx = vctx.borrow_mut();
            fc.begin_region([0.7, 0.7, 0.1, 1.0], || "vctx.prepass_draw");
            vctx.prepass_draw(&mut fc, &world);
            fc.end_region();
            fc.begin_region([0.5, 0.5, 0.5, 1.0], || "gui.prepass_draw");
            let gui = guictx.prepass_draw(&mut fc);
            gui.push_cmd(GuiOrderedCmd {
                z_index: GUI_Z_LAYER_BACKGROUND,
                color: GUI_WHITE,
                cmd: GuiCmd::Rectangle {
                    style: GuiControlStyle::Window,
                    rect: GuiRect::from_xywh((0.0, 5.0), (0.0, 5.0), (0.0, 400.0), (0.0, 300.0)),
                },
            });
            gui.push_cmd(GuiOrderedCmd {
                z_index: GUI_Z_LAYER_BACKGROUND + GUI_Z_OFFSET_CONTROL,
                color: GUI_BLACK,
                cmd: GuiCmd::FreeText {
                    text: Cow::from(format!(
                        "{}\nLookV: {:?}\nLookP: {:?}\nPlace dir: {} {:?}\n{:?}\n",
                        DEBUG_DATA.hud_format(),
                        look_pos,
                        look_precise_pos,
                        i_orientation,
                        OctahedralOrientation::from_index(i_orientation as usize).unwrap(),
                        OctahedralOrientation::from_index(i_orientation as usize)
                            .unwrap()
                            .to_matrixi(),
                    )),
                    scale: 0.5,
                    start_at: gv2((0.0, 10.0), (0.0, 10.0)),
                },
            });
            gui.push_cmd(GuiOrderedCmd {
                z_index: GUI_Z_LAYER_BACKGROUND + GUI_Z_OFFSET_CONTROL,
                color: GUI_WHITE,
                cmd: GuiCmd::Rectangle {
                    style: GuiControlStyle::Crosshair,
                    rect: GuiControlStyle::Crosshair
                        .gui_rect_centered(GuiVec2(GuiCoord(0.5, 0.0), GuiCoord(0.5, 0.0))),
                },
            });
            {
                let bsize = 48.0;
                let bgap = 10.0;
                let mut x = -(i_placeable.len() as f32) / 2.0 * (bsize + bgap);
                let y = -bsize - bgap - 8.0;
                for (i, &def) in i_placeable.iter().enumerate() {
                    gui.push_cmd(GuiOrderedCmd {
                        z_index: GUI_Z_LAYER_HUD + GUI_Z_OFFSET_CONTROL,
                        color: GUI_WHITE,
                        cmd: GuiCmd::VoxelPreview {
                            texture: def.texture_mapping.clone(),
                            rect: GuiRect::from_xywh(
                                (0.5, x + bgap / 2.0),
                                (1.0, y),
                                (0.0, bsize),
                                (0.0, bsize),
                            ),
                        },
                    });
                    gui.push_cmd(GuiOrderedCmd {
                        z_index: GUI_Z_LAYER_HUD + GUI_Z_OFFSET_BG,
                        color: GUI_WHITE,
                        cmd: GuiCmd::Rectangle {
                            style: if i == i_place {
                                GuiControlStyle::Window
                            } else {
                                GuiControlStyle::Button
                            },
                            rect: GuiRect::from_xywh(
                                (0.5, x),
                                (1.0, y - bgap / 2.0),
                                (0.0, bsize + bgap),
                                (0.0, bsize + bgap),
                            ),
                        },
                    });
                    x += bsize + bgap;
                }
            }
            fc.end_region();
            let mut fc = RenderingContext::frame_goto_pass(fc);
            fc.begin_region([0.3, 0.3, 0.8, 1.0], || "vctx.inpass_draw");
            vctx.inpass_draw(&mut fc, &world);
            fc.end_region();

            fc.begin_region([0.5, 0.5, 0.5, 1.0], || "gui.inpass_draw");
            guictx.inpass_draw(&mut fc);
            fc.end_region();

            let fc = RenderingContext::frame_goto_postpass(fc);
            fc.insert_label([0.1, 0.8, 0.1, 1.0], || "frame_finish");
            RenderingContext::frame_finish(fc);
        }

        input_mgr.pre_process();
        for event in event_pump.poll_iter() {
            input_mgr.process(&mut rctx, event);
        }
        input_mgr.post_events_update(&mut rctx);
        i_place = ((-input_mgr.input_state.scroller % i_placeable.len() as i32
            + i_placeable.len() as i32) as usize)
            % i_placeable.len();

        if input_mgr.input_state.requesting_exit {
            input_mgr.input_state.requesting_exit = false;
            break 'running;
        }

        if input_mgr
            .just_pressed_keys
            .contains(&sdl2::keyboard::Keycode::Q)
        {
            i_orientation = (i_orientation + 1) % 24;
        } else if input_mgr
            .just_pressed_keys
            .contains(&sdl2::keyboard::Keycode::E)
        {
            i_orientation = (i_orientation + 23) % 24;
        }

        let player_pos;
        let player_ang;
        {
            let entities = world.ecs();
            let lp_loc: &CLocation = entities.get_component(client_world.local_player).unwrap();
            player_pos = lp_loc.position;
            player_ang = lp_loc.orientation;
            DEBUG_DATA
                .local_player_x
                .store((lp_loc.position.x * 10.0) as i64, Ordering::Release);
            DEBUG_DATA
                .local_player_y
                .store((lp_loc.position.y * 10.0) as i64, Ordering::Release);
            DEBUG_DATA
                .local_player_z
                .store((lp_loc.position.z * 10.0) as i64, Ordering::Release);

            let primary = input_mgr.input_state.primary_action.get_and_reset_pressed();
            let secondary = input_mgr
                .input_state
                .secondary_action
                .get_and_reset_pressed();
            use bxw_world::raycast;
            let mview = glm::quat_to_mat3(&player_ang).transpose();
            let fwd = mview * vec3(0.0, 0.0, 1.0);
            let rc = raycast::RaycastQuery::new_directed(
                player_pos + vec3(0.0, PLAYER_EYE_HEIGHT / 2.0, 0.0),
                fwd,
                32.0,
                &world,
                true,
                false,
            )
            .execute();
            if let raycast::Hit::Voxel { position, .. } = &rc.hit {
                look_pos = *position;
                look_precise_pos = rc.hit_point;
            }
            if primary | secondary {
                click_place = secondary;
                if let raycast::Hit::Voxel {
                    position,
                    datum,
                    normal,
                    normal_datum,
                } = &rc.hit
                {
                    if !click_place {
                        click_pos = Some(*position);
                        click_datum = *datum;
                    } else if normal_datum
                        .map(|d| vxreg.get_definition_from_datum(d).selection_shape.is_none())
                        .unwrap_or(false)
                    {
                        let place_pos = *position + normal.to_vec();
                        let player_aabb = lp_loc
                            .bounding_shape
                            .aabb(lp_loc.position)
                            .inflate(-bxw_world::physics::TOUCH_EPSILON);
                        let voxel_aabb = i_placeable[i_place]
                            .collision_shape
                            .unwrap_or_default()
                            .translate(place_pos.map(|c| c as f64));
                        let intersecting = AABB::intersection(player_aabb, voxel_aabb).is_some();
                        if !intersecting {
                            click_pos = Some(place_pos);
                            click_datum = normal_datum.unwrap();
                        }
                    }
                }
            }
        }

        if let Some(fps) = cfg.render_fps_lock {
            let end_current_frame_time = sdl_timer.performance_counter() as f64 * pf_mult;
            let target_ft = 1.0 / f64::from(fps);
            let ms = (target_ft - end_current_frame_time + current_frame_time) * 1000.0;
            if ms > 0.0 {
                sdl_timer.delay(ms as u32);
            }
        }
    }

    drop(world);
    drop(task_pool);
    let vctx = Rc::try_unwrap(vctx)
        .ok()
        .expect("Remaining references to VoxelRenderer")
        .into_inner();
    vctx.destroy(&rctx.handles);
    guictx.destroy(&rctx.handles);
    Arc::try_unwrap(rres)
        .unwrap_or_else(|_| panic!("Handle still held to resource manager"))
        .destroy(&rctx.handles);
    rctx.destroy();
}
