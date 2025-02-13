# Run benchmark

timestamp=$(date +%s)
suite="beam_$timestamp"

helm-run --run-entries mmlu:subject=philosophy,model=simple/model1 --suite $suite --max-eval-instances 10
                     # gsm:model=text_code,follow_format_instructions=instruct
helm-run --run-entries gsm:model=simple/model1,follow_format_instructions=instruct --suite $suite --max-eval-instances 10
# gsm:model=text_code,follow_format_instructions=instruct
# Summarize benchmark results
helm-summarize --suite $suite

# Start a web server to display benchmark results
helm-server --suite $suite
