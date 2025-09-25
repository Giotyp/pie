use inferlet::traits::Tokenize;
use inferlet::{Sampler, set_return, traits::Forward};
use pico_args::Arguments;
use std::ffi::OsString;
use std::time::Instant;

const SYSTEM_PROMPT: &str = r#"<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant. Answer questions clearly and concisely.

"#;

/// Defines the command-line interface and help message.
const HELP: &str = r#"
Usage: memory-swap [OPTIONS]

A simple example demonstrating evict_kv_pages() and restore_kv_pages() functions.

Options:
    -p, --prompt <STRING>      The user prompt to send to the model
                                                        (default: "Hello! Tell me about artificial intelligence.")
    -n, --max-tokens <INT>     The maximum number of new tokens to generate per step
                                                        (default: 30)
    -h, --help                Print help information
"#;

#[inferlet::main]
async fn main() -> Result<(), String> {
    // 1. Get arguments from the inferlet environment and prepare the parser.
    let mut args = Arguments::from_vec(
        inferlet::get_arguments()
            .into_iter()
            .map(OsString::from)
            .collect(),
    );

    // 2. Handle the --help flag.
    if args.contains(["-h", "--help"]) {
        println!("{}", HELP);
        return Ok(());
    }

    // 3. Parse arguments, falling back to defaults if they are not provided.
    let prompt = args
        .opt_value_from_str(["-p", "--prompt"])
        .map_err(|e| e.to_string())?
        .unwrap_or_else(|| "Hello! Tell me about artificial intelligence.".to_string());

    let max_tokens: usize = args
        .opt_value_from_str(["-n", "--max-tokens"])
        .map_err(|e| e.to_string())?
        .unwrap_or(30);

    // Ensure no unknown arguments were passed.
    let remaining = args.finish();
    if !remaining.is_empty() {
        return Err(format!(
            "Unknown arguments found: {:?}. Use --help for usage.",
            remaining
        ));
    }

    println!("Simple Memory Swap Test");
    println!("==========================");
    println!("This example demonstrates evict_kv_pages() and restore_kv_pages() functions");
    println!("Using prompt: {}", prompt);
    println!("Max tokens per generation: {}", max_tokens);
    println!();

    test_memory_swap(&prompt, max_tokens).await
}

async fn test_memory_swap(prompt: &str, max_tokens: usize) -> Result<(), String> {
    let start = Instant::now();
    let model = inferlet::get_auto_model();
    let _tokenizer = model.get_tokenizer();

    // Create a context and fill it with some content
    println!("1. Creating context and filling with initial content...");
    let mut ctx = model.create_context();
    ctx.fill_system(SYSTEM_PROMPT);
    ctx.fill_user(&prompt);

    println!("   Context created with {} KV pages", ctx.kv_pages.len());

    // Generate some initial response to build up KV cache
    println!("\n2. Generating initial response to build KV cache...");
    let final_text = ctx
        .generate_until(Sampler::top_p(0.6, 0.95), max_tokens as usize)
        .await;
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

    Ok(())
}
