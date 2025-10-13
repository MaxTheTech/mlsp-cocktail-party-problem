import torchaudio

# List all available dataset classes
print("Available torchaudio datasets:")
for ds in dir(torchaudio.datasets) :
    print(f"  - {ds}")