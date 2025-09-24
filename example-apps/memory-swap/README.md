# Simple Memory Swap Example

This example demonstrates the basic memory swapping functionality in PIE, specifically the `evict_kv_pages()` and `restore_kv_pages()` functions. These functions enable efficient memory management by temporarily moving KV cache pages to storage and restoring them when needed.

## Overview

This simplified example shows:
1. Creating a context and building up KV cache
2. Evicting KV pages to free memory
3. Simulating memory pressure
4. Restoring KV pages from storage
5. Verifying context integrity after restoration

## Key Functions Demonstrated

### `evict_kv_pages(kv_pages: &[KvPage], name: &str)`
- **Purpose**: Moves KV cache pages from GPU memory to storage
- **Use Case**: Free up memory when system is under pressure
- **Parameters**: 
  - `kv_pages`: Array of KV cache pages to evict
  - `name`: Unique identifier for the swapped context

### `restore_kv_pages(name: &str)`
- **Purpose**: Restores previously evicted KV cache pages back to GPU memory
- **Use Case**: Resume inference with a previously swapped context
- **Parameters**:
  - `name`: Identifier of the context to restore

## Usage

```bash
# Run the simple memory swap test
cargo run --package memory-swap
```

## Expected Output

```
🔄 Simple Memory Swap Test
==========================
This example demonstrates evict_kv_pages() and restore_kv_pages() functions

1️⃣ Creating context and filling with initial content...
   ✅ Context created with 0 KV pages

2️⃣ Generating initial response to build KV cache...
   ✅ Generated: Artificial intelligence (AI) refers to computer systems...
   📊 KV cache now has 15 pages

3️⃣ Evicting KV pages to storage...
   ✅ KV pages evicted to storage key: 'my_conversation'

4️⃣ Simulating memory pressure (creating new context)...
   ⚠️  Memory pressure applied

5️⃣ Restoring KV pages from storage...
   ✅ KV pages restored from storage

6️⃣ Testing context integrity by continuing conversation...
   ✅ Continued conversation: For example, machine learning algorithms...

🎉 Memory swap test completed successfully!
   The context maintained its state through eviction and restoration.
```

## Implementation Details

### Context Creation and Management
```rust
let mut ctx = model.create_context();
ctx.fill(SYSTEM_PROMPT);
ctx.flush().await;
```

### Memory Eviction
```rust
ctx.queue().evict_kv_pages(&ctx.kv_pages, swap_name);
```

### Memory Restoration
```rust
ctx.queue().restore_kv_pages(swap_name);
```

### Integrity Verification
After restoration, the example continues the conversation to verify that the context state has been preserved correctly.

## Benefits Demonstrated

1. **Memory Efficiency**: Shows how to free up GPU memory when needed
2. **Context Preservation**: Demonstrates that swapped contexts maintain their state
3. **Simple API**: Easy-to-use functions for memory management
4. **Seamless Operation**: Context can be used normally after restoration

## Practical Applications

The memory swapping functionality is useful for:
- **Multi-tenant LLM Serving**: Managing memory across multiple users
- **Long Conversation Support**: Handling extended chat sessions
- **Batch Processing**: Efficient memory usage for large workloads
- **Priority-based Scheduling**: Giving high-priority requests immediate memory access