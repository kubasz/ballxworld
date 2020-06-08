use smallvec::alloc::fmt::Formatter;
use std::fmt::Debug;
use std::iter::FilterMap;
use std::ops::{Index, IndexMut};
use std::slice::*;

pub struct SparseVec<T> {
    elements: Vec<Option<T>>,
    free_list: Vec<usize>,
}

pub type SparseVecIter<'v, T> = FilterMap<Iter<'v, Option<T>>, fn(&Option<T>) -> Option<&T>>;
pub type SparseVecIterMut<'v, T> =
    FilterMap<IterMut<'v, Option<T>>, fn(&mut Option<T>) -> Option<&mut T>>;

impl<T> Default for SparseVec<T> {
    fn default() -> Self {
        Self {
            elements: Default::default(),
            free_list: Default::default(),
        }
    }
}

impl<T: Clone> Clone for SparseVec<T> {
    fn clone(&self) -> Self {
        Self {
            elements: self.elements.clone(),
            free_list: self.free_list.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.elements.clone_from(&source.elements);
        self.free_list.clone_from(&source.free_list);
    }
}

impl<T: Debug> Debug for SparseVec<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sparse")?;
        Debug::fmt(&self.elements, f)
    }
}

impl<T> SparseVec<T> {
    pub fn add(&mut self, value: T) -> usize {
        let idx = if let Some(idx) = self.free_list.pop() {
            idx
        } else {
            let old_len = self.elements.len();
            let new_len = 2 * old_len + 1;
            self.elements.resize_with(new_len, || None);
            self.free_list.extend(old_len + 1..new_len);
            old_len
        };
        self.elements[idx] = Some(value);
        idx
    }

    pub fn remove(&mut self, index: usize) -> T {
        self.free_list.push(index);
        self.elements[index].take().unwrap()
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.elements.get(index).and_then(Option::as_ref)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.elements.get_mut(index).and_then(Option::as_mut)
    }

    pub fn len(&self) -> usize {
        self.elements.len() - self.free_list.len()
    }

    pub fn is_empty(&self) -> bool {
        self.elements.len() == self.free_list.len()
    }

    pub fn capacity(&self) -> usize {
        self.elements.len()
    }

    pub fn iter(&self) -> SparseVecIter<T> {
        self.elements.iter().filter_map(|x| x.as_ref())
    }

    pub fn iter_mut(&mut self) -> SparseVecIterMut<T> {
        self.elements.iter_mut().filter_map(|x| x.as_mut())
    }
}

impl<T> Index<usize> for SparseVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("Index out of bounds for SparseVec")
    }
}

impl<T> IndexMut<usize> for SparseVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index)
            .expect("Index out of bounds for SparseVec")
    }
}
