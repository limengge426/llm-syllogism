import os
import json
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from asyncio import Semaphore, Lock as AsyncLock 
from tqdm import tqdm
import random

load_dotenv()

MODELS_TO_TEST = [
    {
        "model_name": "qwen3-max",
        "api_key": os.getenv("QWEN_API_KEY"),
        "base_url": os.getenv("QWEN_BASE_URL"),
        "max_workers": 8,
        "batch_size": 200,
        "retry_count": 3,
    },
    {
        "model_name": "deepseek-chat",
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL"),
        "max_workers": 20,
        "batch_size": 200,
        "retry_count": 3,
    },
    {
        "model_name": "gemini-2.0-flash",
        "api_key": os.getenv("GEMINI_API_KEY"),
        "base_url": os.getenv("GEMINI_BASE_URL"),
        "max_workers": 20,
        "batch_size": 200,
        "retry_count": 3,
    },
        {
        "model_name": "llama-3.3-70b-versatile",
        "api_key": os.getenv("LLAMA_API_KEY"),
        "base_url": os.getenv("LLAMA_BASE_URL"),
        "max_workers": 15,
        "batch_size": 200,
        "retry_count": 3,
    },
]

write_lock = AsyncLock()


async def async_ask_with_retry(client: AsyncOpenAI, model_name: str, prompt: str, retry_count: int, semaphore: Semaphore):
    
    async with semaphore:
        for attempt in range(retry_count):
            try:
                resp = await client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=150,
                )
                text = resp.choices[0].message.content.strip()
                
                try:
                    data = json.loads(text)
                except:
                    data = {"valid": None, "why": text}
                return data, "success"
                
            except Exception as e:
                error_msg = str(e)
                
                if "429" in error_msg or "rate" in error_msg.lower():
                    wait_time = (2 ** attempt) * 2
                    await asyncio.sleep(wait_time)
                else:
                    if attempt < retry_count - 1:
                        await asyncio.sleep(2 ** attempt)
                
                if attempt == retry_count - 1:
                    return {"valid": None, "why": f"Error: {error_msg}"}, "failed"
        
    return {"valid": None, "why": "Max retries exceeded"}, "failed"


async def async_process_single_task(args):

    rec["model"] = model_name
    
    result, status = await async_ask_with_retry(
        client, 
        model_name, 
        rec["prompt"],
        retry_count=config.get("retry_count", 3),
        semaphore=semaphore
    )
    rec["reply"] = result
    return rec, status


async def async_process_batch(client: AsyncOpenAI, model_name: str, prompts: list, output_file: Path, config: dict, batch_info: tuple):

    batch_num, num_batches = batch_info
    max_workers = config.get("max_workers", 10)
    semaphore = Semaphore(max_workers)
    
    results = []
    failed_count = 0
    success_count = 0
    
    tasks_args = [(client, model_name, rec.copy(), config, semaphore) for rec in prompts]
    tasks = [async_process_single_task(args) for args in tasks_args]
    
    print(f"--- Batch {batch_num}/{num_batches} (Concurrency: {max_workers}) ---")
    with tqdm(total=len(tasks), desc=f"Batch {batch_num}/{num_batches}", unit="req") as pbar:
        for future in asyncio.as_completed(tasks):
            try:
                result, status = await future
                results.append(result)
                
                if status == "success":
                    success_count += 1
                else:
                    failed_count += 1
                
                pbar.update(1)
                
            except Exception:
                failed_count += 1
                pbar.update(1)
    
    async with write_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    return success_count, failed_count


async def async_main(inp="data/prompts_wordnet.jsonl", out_dir="data"):
    
    lines = Path(inp).read_text(encoding="utf-8").strip().splitlines()

    prompts = [json.loads(line) for line in lines]
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    
    print(f"Loaded {len(prompts)} total tasks from {inp}\n")
    
    for config in MODELS_TO_TEST:
        model_name = config["model_name"]
        api_key = config["api_key"]
        base_url = config.get("base_url")
        
        if not api_key:
            print(f"Skip {model_name} (no API key)\n")
            continue
        
        max_workers_display = config.get('max_workers', 10)
        batch_size_display = config.get('batch_size', 200)

        print(f"--- Starting Model: {model_name} ---")
        print(f"Workers (Concurrency): {max_workers_display}, Batch Size: {batch_size_display}")
        
        start_time = time.time()
        
        output_file = Path(out_dir) / f"results_{model_name}.jsonl"
        
        processed_ids = set()
        if output_file.exists():
            print(f"  Found existing file: {output_file}. Reading processed IDs...")
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if 'id' in data:
                                processed_ids.add(data['id'])
                        except json.JSONDecodeError:
                            pass 
            except Exception as e:
                print(f"  Warning: Could not read existing file. Will start from scratch. Error: {e}")
                processed_ids = set()
        
        if processed_ids:
            print(f"  Found {len(processed_ids)} already processed prompts.")
        else:
            print(f"  No existing results found. Starting from scratch.")
        
        prompts_to_run = [p for p in prompts if p.get('id') not in processed_ids]
        
        if not prompts_to_run:
            print(f"  All {len(prompts)} prompts already processed for {model_name}. Skipping.\n" + "="*40 + "\n")
            continue
        
        print(f"  {len(prompts_to_run)} new prompts to process for {model_name}.")
        
        total_success = 0
        total_failed = 0
        batch_size = config.get("batch_size", 200)
        
        try:
            async with AsyncOpenAI(api_key=api_key, base_url=base_url) if base_url else AsyncOpenAI(api_key=api_key) as client:
                
                num_batches = (len(prompts_to_run) + batch_size - 1) // batch_size
                
                for batch_num, batch_start in enumerate(range(0, len(prompts_to_run), batch_size), 1):
                    batch_end = min(batch_start + batch_size, len(prompts_to_run))
                    batch_prompts = prompts_to_run[batch_start:batch_end]
                    
                    success, failed = await async_process_batch(
                        client, model_name, batch_prompts,
                        output_file, config, (batch_num, num_batches)
                    )
                    
                    total_success += success
                    total_failed += failed
                    
                    if batch_end < len(prompts_to_run):
                        await asyncio.sleep(2)
            
            await asyncio.sleep(0.1)
            
            elapsed = time.time() - start_time
            total_processed = total_success + total_failed
            
            print(f"\n--- Completed processing for: {model_name} ---")
            print(f"New Success: {total_success}/{total_processed}")
            print(f"Total processed for this run: {total_processed}")
            print(f"Time: {elapsed/60:.1f} min")
            print(f"Output: {output_file}\n" + "="*40 + "\n")
            
        except Exception as e:
            print(f"\nError during processing for {model_name}: {e}\n")
            continue


if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nAn unexpected error occurred in main: {e}")

