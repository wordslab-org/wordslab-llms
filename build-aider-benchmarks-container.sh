# Clone the aider repo
git clone https://github.com/Aider-AI/aider.git

# Create the scratch dir to hold benchmarking results inside the main aider dir:
cd aider
mkdir tmp.benchmarks

# Clone the repo with the exercises
git clone https://github.com/exercism/python tmp.benchmarks/python-benchmark/python
git clone https://github.com/exercism/csharp tmp.benchmarks/csharp-benchmark/csharp
git clone https://github.com/Aider-AI/polyglot-benchmark tmp.benchmarks/polyglot-benchmark

# Build container
docker build --file benchmark/Dockerfile -t aider-benchmark .