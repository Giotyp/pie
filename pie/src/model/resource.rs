use crate::instance::InstanceId;
use crate::runtime::{TerminationCause, trap};
use crate::utils::IdPool;
use std::collections::{HashMap, HashSet, hash_map::Entry};
use std::time::Instant;
use thiserror::Error;
pub type ResourceId = u32;
pub type ResourceTypeId = u32;
pub type DeviceId = usize;

pub static KV_PAGE_TYPE_ID: ResourceTypeId = 0;
pub static EMBED_TYPE_ID: ResourceTypeId = 1;
pub static ADAPTER_TYPE_ID: ResourceTypeId = 2;

pub static GPU_0: DeviceId = 0;
pub static CPU_0: DeviceId = 1;

pub trait DeviceIdExt {
    fn is_gpu(&self) -> bool;
    fn get_gpu() -> DeviceId {
        GPU_0
    }
    fn get_cpu() -> DeviceId {
        CPU_0
    }
}

impl DeviceIdExt for DeviceId {
    fn is_gpu(&self) -> bool {
        // Start definition with GPUs
        *self < CPU_0
    }
}
// ---- Custom ResourceError enum ----
#[derive(Debug, Error)]
pub enum ResourceError {
    #[error("Resource pool for type {type_id:?} does not exist")]
    PoolNotFound { type_id: ResourceTypeId },

    #[error("Out of memory for resource type {type_id:?}")]
    OutOfMemory { type_id: ResourceTypeId },

    #[error("Instance {inst_id:?} has no allocated resources of type {type_id:?}")]
    InstanceNotAllocated {
        inst_id: InstanceId,
        type_id: ResourceTypeId,
    },

    #[error("Pointer {ptr:?} is not allocated to instance {inst_id:?}")]
    PointerNotAllocated {
        ptr: ResourceId,
        inst_id: InstanceId,
    },

    #[error("Exported resource with name '{name}' already exists")]
    ExportNameExists { name: String },

    #[error("Exported resource with name '{name}' not found")]
    ExportNotFound { name: String },

    #[error("OOM unrecoverable: {0}")]
    OomUnrecoverable(String),

    #[error("IdPool error: {0:?}")]
    IdPoolError(String),

    #[error("Multiple errors occurred: GPU error: {gpu_error}, CPU error: {cpu_error}")]
    MultipleErrors {
        gpu_error: Box<ResourceError>,
        cpu_error: Box<ResourceError>,
    },
}

/// Manages the state of all resources, instances, and exports.
#[derive(Debug)]
pub struct ResourceManager {
    gpu_res_pool: HashMap<ResourceTypeId, IdPool<u32>>,
    cpu_res_pool: HashMap<ResourceTypeId, IdPool<u32>>,
    res_exported: HashMap<ResourceTypeId, HashMap<String, Vec<ResourceId>>>,
    gpu_res_allocated: HashMap<(ResourceTypeId, InstanceId), HashSet<ResourceId>>,
    cpu_res_allocated: HashMap<(ResourceTypeId, InstanceId), HashSet<ResourceId>>,
    inst_start_time: HashMap<InstanceId, Instant>,
}

impl ResourceManager {
    pub fn new(resources: HashMap<ResourceTypeId, u32>) -> Self {
        let mut gpu_res_pool = HashMap::new();
        let mut cpu_res_pool = HashMap::new();
        for (res_id, capacity) in resources {
            gpu_res_pool.insert(res_id, IdPool::new(capacity));
            cpu_res_pool.insert(res_id, IdPool::new(capacity));
        }

        Self {
            gpu_res_pool,
            cpu_res_pool,
            res_exported: HashMap::new(),
            gpu_res_allocated: HashMap::new(),
            cpu_res_allocated: HashMap::new(),
            inst_start_time: HashMap::new(),
        }
    }

    /// A new combined allocation method that handles the OOM logic internally.
    pub fn allocate_with_oom(
        &mut self,
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        count: usize,
    ) -> Result<(Vec<ResourceId>, DeviceId), ResourceError> {
        let available_device: DeviceId;
        // Check GPU first
        if self.available(type_id, DeviceId::get_gpu())? >= count {
            available_device = DeviceId::get_gpu();
        } else if self.available(type_id, DeviceId::get_cpu())? >= count {
            available_device = DeviceId::get_cpu();
        } else {
            // Not enough memory, trigger the OOM killer for GPU
            self.oom_kill(type_id, count, inst_id, DeviceId::get_gpu())?;
            available_device = DeviceId::get_gpu();
        }

        // A successful oom_kill guarantees enough space.
        self.allocate(inst_id, type_id, available_device, count)
    }

    pub fn evict_restore(
        &mut self,
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        device_ids: Vec<DeviceId>,
        ptrs: Vec<ResourceId>,
        evict: bool,
    ) -> Result<(Vec<ResourceId>, DeviceId), ResourceError> {
        // Deallocate pointers from GPU pool
        let mut dealloc_devs = Vec::new();
        let mut dealloc_ptrs = Vec::new();
        for (ptr, device_id) in ptrs.iter().zip(device_ids.iter()) {
            if evict && !device_id.is_gpu() {
                // CPU resources are not evicted
                continue;
            } else if !evict && device_id.is_gpu() {
                // GPU resources are not restored
                continue;
            }
            dealloc_devs.push(*device_id);
            dealloc_ptrs.push(*ptr);
        }

        let count = dealloc_ptrs.len();
        self.deallocate(inst_id, type_id, dealloc_ptrs, dealloc_devs)?;

        let device_id = if evict {
            DeviceId::get_cpu()
        } else {
            DeviceId::get_gpu()
        };

        if self.available(type_id, device_id)? < count {
            self.oom_kill(type_id, count, inst_id, device_id)?;
        }

        // Return new physical pointers
        self.allocate(inst_id, type_id, device_id, count)
    }

    fn available(
        &self,
        type_id: ResourceTypeId,
        device_id: DeviceId,
    ) -> Result<usize, ResourceError> {
        let res_pool = {
            if device_id.is_gpu() {
                &self.gpu_res_pool
            } else {
                &self.cpu_res_pool
            }
        };

        let pool = res_pool
            .get(&type_id)
            .ok_or(ResourceError::PoolNotFound { type_id })?;
        Ok(pool.available())
    }

    fn allocate(
        &mut self,
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        device_id: DeviceId,
        count: usize,
    ) -> Result<(Vec<ResourceId>, DeviceId), ResourceError> {
        let res_pool = {
            if device_id.is_gpu() {
                &mut self.gpu_res_pool
            } else {
                &mut self.cpu_res_pool
            }
        };
        let res_allocated = {
            if device_id.is_gpu() {
                &mut self.gpu_res_allocated
            } else {
                &mut self.cpu_res_allocated
            }
        };

        let pool = res_pool
            .get_mut(&type_id)
            .ok_or(ResourceError::PoolNotFound { type_id })?;

        if pool.available() < count {
            return Err(ResourceError::OutOfMemory { type_id });
        }

        let allocated = pool.acquire_many(count).unwrap();
        self.inst_start_time
            .entry(inst_id)
            .or_insert_with(Instant::now);
        res_allocated
            .entry((type_id, inst_id))
            .or_default()
            .extend(&allocated);

        Ok((allocated, device_id))
    }

    pub fn deallocate(
        &mut self,
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        ptrs: Vec<ResourceId>,
        device_ids: Vec<DeviceId>,
    ) -> Result<(), ResourceError> {
        match self.deallocate_device(inst_id, type_id, device_ids, &ptrs) {
            Ok(()) => return Ok(()),
            Err(error) => {
                return Err(error);
            }
        }
    }

    fn deallocate_device(
        &mut self,
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        device_ids: Vec<DeviceId>,
        ptrs: &Vec<ResourceId>,
    ) -> Result<(), ResourceError> {
        for (ptr, device_id) in ptrs.iter().zip(device_ids.iter()) {
            let res_allocated = {
                if device_id.is_gpu() {
                    &mut self.gpu_res_allocated
                } else {
                    &mut self.cpu_res_allocated
                }
            };
            let res_pool = {
                if device_id.is_gpu() {
                    &mut self.gpu_res_pool
                } else {
                    &mut self.cpu_res_pool
                }
            };

            let allocated = res_allocated
                .get_mut(&(type_id, inst_id))
                .ok_or(ResourceError::InstanceNotAllocated { inst_id, type_id })?;

            let pool = res_pool
                .get_mut(&type_id)
                .ok_or(ResourceError::PoolNotFound { type_id })?;

            if allocated.remove(&ptr) {
                pool.release(*ptr).unwrap();
            }
        }

        Ok(())
    }

    fn oom_kill(
        &mut self,
        type_id: ResourceTypeId,
        size: usize,
        inst_id_to_exclude: InstanceId,
        device_id: DeviceId,
    ) -> Result<(), ResourceError> {
        let requester_start_time = self
            .inst_start_time
            .get(&inst_id_to_exclude)
            .copied()
            .ok_or_else(|| {
                ResourceError::OomUnrecoverable(
                    "Requesting instance has no start time.".to_string(),
                )
            })?;

        loop {
            if self.available(type_id, device_id)? >= size {
                break;
            }

            let victim_id = self
                .inst_start_time
                .iter()
                .filter(|(id, time)| **id != inst_id_to_exclude && **time > requester_start_time)
                .max_by_key(|(_, time)| **time)
                .map(|(id, _)| *id);

            if let Some(victim_id) = victim_id {
                self.cleanup_id(victim_id, device_id)?;
                trap(
                    victim_id,
                    TerminationCause::OutOfResources(
                        "Terminated by OOM killer for an older instance".to_string(),
                    ),
                );
            } else {
                return Err(ResourceError::OomUnrecoverable(
                    "Not enough memory after terminating all newer instances.".to_string(),
                ));
            }
        }
        Ok(())
    }

    pub fn cleanup(&mut self, inst_id: InstanceId) -> Result<(), ResourceError> {
        // Clean up both GPU and CPU resources
        let gpu_cleanup = self.cleanup_id(inst_id, GPU_0);
        let cpu_cleanup = self.cleanup_id(inst_id, CPU_0);

        match (gpu_cleanup, cpu_cleanup) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(gpu_error), Err(cpu_error)) => Err(ResourceError::MultipleErrors {
                gpu_error: Box::new(gpu_error),
                cpu_error: Box::new(cpu_error),
            }),
            (Err(gpu_error), _) => Err(gpu_error),
            (_, Err(cpu_error)) => Err(cpu_error),
        }
    }

    fn cleanup_id(
        &mut self,
        inst_id: InstanceId,
        device_id: DeviceId,
    ) -> Result<(), ResourceError> {
        let mut to_release_by_type: HashMap<ResourceTypeId, Vec<ResourceId>> = HashMap::new();

        let res_allocated = {
            if device_id.is_gpu() {
                &mut self.gpu_res_allocated
            } else {
                &mut self.cpu_res_allocated
            }
        };
        let res_pool = {
            if device_id.is_gpu() {
                &mut self.gpu_res_pool
            } else {
                &mut self.cpu_res_pool
            }
        };

        res_allocated.retain(|(ty, id), ptrs| {
            if *id == inst_id {
                to_release_by_type
                    .entry(*ty)
                    .or_default()
                    .extend(ptrs.iter());
                false
            } else {
                true
            }
        });

        for (ty, ptrs) in to_release_by_type {
            let pool = res_pool
                .get_mut(&ty)
                .ok_or(ResourceError::PoolNotFound { type_id: ty })?;
            for ptr in ptrs {
                pool.release(ptr).unwrap();
            }
        }
        self.inst_start_time.remove(&inst_id);
        Ok(())
    }

    // --- export, import, release_exported, and get_all_exported methods ---
    // These are moved here from Model with minimal changes, now returning ResourceError
    pub fn export(
        &mut self,
        inst_id: InstanceId,
        type_id: ResourceTypeId,
        ptrs: Vec<ResourceId>,
        name: String,
    ) -> Result<(), ResourceError> {
        let allocated = self
            .gpu_res_allocated
            .get_mut(&(type_id, inst_id))
            .ok_or(ResourceError::InstanceNotAllocated { inst_id, type_id })?;

        for ptr in &ptrs {
            if !allocated.contains(ptr) {
                return Err(ResourceError::PointerNotAllocated { ptr: *ptr, inst_id });
            }
        }

        let type_exports = self.res_exported.entry(type_id).or_default();
        match type_exports.entry(name) {
            Entry::Occupied(entry) => Err(ResourceError::ExportNameExists {
                name: entry.key().clone(),
            }),
            Entry::Vacant(entry) => {
                ptrs.iter().for_each(|ptr| {
                    allocated.remove(ptr);
                });
                entry.insert(ptrs);
                Ok(())
            }
        }
    }

    pub fn import(
        &mut self,
        type_id: ResourceTypeId,
        name: String,
    ) -> Result<Vec<ResourceId>, ResourceError> {
        self.res_exported
            .get(&type_id)
            .and_then(|exports| exports.get(&name))
            .cloned()
            .ok_or(ResourceError::ExportNotFound { name })
    }

    pub fn release_exported(
        &mut self,
        type_id: ResourceTypeId,
        name: String,
    ) -> Result<(), ResourceError> {
        let type_exports = self
            .res_exported
            .get_mut(&type_id)
            .ok_or(ResourceError::PoolNotFound { type_id })?;

        if let Some(ptrs_to_release) = type_exports.remove(&name) {
            let pool = self
                .gpu_res_pool
                .get_mut(&type_id)
                .ok_or(ResourceError::PoolNotFound { type_id })?;
            for ptr in ptrs_to_release {
                pool.release(ptr).unwrap();
            }
            Ok(())
        } else {
            Err(ResourceError::ExportNotFound { name })
        }
    }

    pub fn get_all_exported(&self, type_id: ResourceTypeId) -> Vec<(String, Vec<ResourceId>)> {
        self.res_exported
            .get(&type_id)
            .map(|exports| {
                exports
                    .iter()
                    .map(|(name, ptrs)| (name.clone(), ptrs.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Appends detailed statistics about the resource manager's state to a given HashMap.
    pub fn append_stats_to(&self, stats: &mut HashMap<String, String>) {
        // Report on each resource pool
        for pool_type in vec![&self.gpu_res_pool, &self.cpu_res_pool] {
            for (type_id, pool) in pool_type {
                let capacity = pool.capacity() as usize;
                let available = pool.available();
                let used = capacity - available;

                stats.insert(
                    format!("resource.{}.capacity", type_id),
                    capacity.to_string(),
                );
                stats.insert(
                    format!("resource.{}.available", type_id),
                    available.to_string(),
                );
                stats.insert(format!("resource.{}.used", type_id), used.to_string());
            }
        }

        // Report on active instances
        stats.insert(
            "instances.active_count".to_string(),
            self.inst_start_time.len().to_string(),
        );

        // Report on exported resources
        for (type_id, exports) in &self.res_exported {
            stats.insert(
                format!("resource.{}.exported_count", type_id),
                exports.len().to_string(),
            );
        }
    }
}
