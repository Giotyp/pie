use inferlet::interface::Forward;
use inferlet::stop_condition::{StopCondition, ends_with_any, max_len};
use inferlet::{Args, Result, Sampler, set_return};
use std::time::Instant;

const SYSTEM_PROMPT: &str = r#"<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant. Answer questions clearly and concisely.

"#;

#[inferlet::main]
async fn main(mut args: Args) -> Result<String> {
    let prompt: String = args.value_from_str(["-p", "--prompt"])?;
    let max_num_outputs: usize = args.value_from_str(["-n", "--max-tokens"]).unwrap_or(256);

    let start = Instant::now();
    let model = inferlet::get_auto_model();

    // Create a context and fill it with some content
    println!("1. Creating context and filling with initial content...");
    let mut ctx = model.create_context();
    ctx.fill_system(SYSTEM_PROMPT);
    ctx.fill_user(&prompt);

    println!("   Context created with {} KV pages", ctx.kv_pages.len());

    // Generate some initial response to build up KV cache
    println!("\n2. Generating initial response to build KV cache...");
    let sampler = Sampler::top_p(0.6, 0.95);
    let stop_cond = max_len(max_num_outputs).or(ends_with_any(model.eos_tokens()));
    let final_text = ctx.generate(sampler, stop_cond).await;
    println!("   Generated: {}", final_text.trim());
    println!("   KV cache now has {} pages", ctx.kv_pages.len());

    // Evict KV pages to free memory
    println!("\n3. Evicting KV pages to storage...");
    ctx.queue().evict_kv_pages(&ctx.kv_pages[0..2]);

    // Restore KV pages from storage
    println!("\n4. Restoring KV pages from storage...");
    ctx.queue().restore_kv_pages(&ctx.kv_pages[0..2]);

    println!("\nMemory swap test completed successfully!");
    println!("   Total time: {:?}", start.elapsed());

    // Set the return value for the inferlet
    set_return(&final_text);

    Ok(final_text)
}
