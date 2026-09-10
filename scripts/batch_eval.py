#!/usr/bin/env python3
import urllib.request
import json
import concurrent.futures
import argparse
import time

def send_request(base_url, req_idx):
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": "moonshotai/Kimi-K3",
        "messages": [
            {"role": "user", "content": f"Question {req_idx}: Explain the significance of multi-rail InfiniBand in distributed AI clusters in exactly 50 words."}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(endpoint, data=data, headers={'Content-Type': 'application/json'})
    
    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            result = json.loads(response.read().decode())
            # Just grab a snippet of the response to verify it worked
            text = result['choices'][0]['message']['content'][:20].replace('\n', ' ')
            return req_idx, True, time.time() - start_time, f"Success (Snippet: '{text}...')", result['usage']
    except Exception as e:
        return req_idx, False, time.time() - start_time, str(e), None

def main():
    parser = argparse.ArgumentParser(description="Stress test SGLang API")
    parser.add_argument("--base-url", required=True, help="e.g., http://c104-020:30000/v1")
    parser.add_argument("--requests", type=int, default=100, help="Total number of requests to send")
    parser.add_argument("--concurrency", type=int, default=32, help="Number of concurrent workers")
    args = parser.parse_args()

    print(f"Starting batch evaluation test on {args.base_url}")
    print(f"Sending {args.requests} requests with a concurrency of {args.concurrency}...\n")

    success_count = 0
    failure_count = 0
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(send_request, args.base_url, i): i for i in range(1, args.requests + 1)}
        
        for future in concurrent.futures.as_completed(futures):
            req_idx, success, duration, msg, usage = future.result()
            if success:
                success_count += 1
                print(f"[Req {req_idx:03d}] \033[92mPASS\033[0m in {duration:.2f}s | {msg}")
            else:
                failure_count += 1
                print(f"[Req {req_idx:03d}] \033[91mFAIL\033[0m in {duration:.2f}s | Error: {msg}")

    total_time = time.time() - start_time
    print("\n" + "="*40)
    print("BATCH EVALUATION RESULTS")
    print("="*40)
    print(f"Total Requests: {args.requests}")
    print(f"Concurrency:    {args.concurrency}")
    print(f"Total Time:     {total_time:.2f}s")
    print(f"Throughput:     {args.requests / total_time:.2f} requests/sec")
    print(f"Successful:     \033[92m{success_count}\033[0m")
    print(f"Failed:         \033[91m{failure_count}\033[0m")

if __name__ == "__main__":
    main()