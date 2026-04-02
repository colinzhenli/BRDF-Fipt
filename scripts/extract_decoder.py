import torch
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Extract only material.decoder weights from a checkpoint.")
    parser.add_argument("input_ckpt", type=str, help="Path to the input checkpoint file (.ckpt)")
    args = parser.parse_args()

    input_path = args.input_ckpt
    if not os.path.isfile(input_path):
        print(f"Error: File not found at {input_path}")
        return

    # Create the output path by appending a postfix
    base_name, ext = os.path.splitext(input_path)
    output_path = f"{base_name}_decoder_only{ext}"

    print(f"Loading checkpoint from: {input_path}")
    # Use mmap=True here as well so it's blazing fast
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)

    if 'state_dict' not in checkpoint:
        print("Error: 'state_dict' not found in the checkpoint.")
        return

    # Extract only the material.decoder.* weights
    new_state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        if 'material.decoder.' in k:
            # We clone the tensor to ensure it is isolated from the memory-mapped original checkpoint
            new_state_dict[k] = v.clone()

    print(f"Extracted {len(new_state_dict)} keys for the decoder.")

    # Substitute the state_dict
    checkpoint['state_dict'] = new_state_dict
    
    # Clean up massive optimizer states or callbacks to ensure the new checkpoint is as tiny as possible
    keys_to_remove = ['optimizer_states', 'lr_schedulers', 'callbacks', 'loops']
    for key in keys_to_remove:
        if key in checkpoint:
            del checkpoint[key]

    print(f"Saving stripped checkpoint to: {output_path}")
    torch.save(checkpoint, output_path)
    print("Done! The new checkpoint is much smaller and faster to load.")

if __name__ == "__main__":
    main()
