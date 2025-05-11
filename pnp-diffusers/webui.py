import gradio as gr
import yaml
import os
from PIL import Image
import subprocess

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_default_config():
    """Create default config file if it doesn't exist"""
    config_path = "config_pnp.yaml"
    if not os.path.exists(config_path):
        default_config = {
            "device": 'cuda',
            "guidance_scale": 7.5,
            "image_path": 'data/original.png',
            "latents_path": 'latents_forward',
            "n_timesteps": 25,
            "negative_prompt": "ugly, blurry, black, low res, unrealistic",
            "output_path": "360PanT-results/original",
            "pnp_attn_t": 0.5,
            "pnp_f_t": 0.8,
            "prompt": "a photo of castle.",
            "sd_version": '2.1',
            "seed": 1,
        }
        with open(config_path, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False)

def run_preprocess(image, method):
    """Run the preprocessing step"""
    # Ensure data directory exists
    ensure_dir("data")
    
    # Save uploaded image temporarily
    image_path = os.path.join("data", "temp_input.png")
    image.save(image_path)
    
    # Run preprocess.py
    command = f"python preprocess.py --method {method} --data_path {image_path} --save-steps 50"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running preprocess: {str(e)}")
        return None
    
    return image_path

def generate_image(image_path, method, prompt):
    """Generate the translated image"""
    config_path = "config_pnp.yaml"
    create_default_config()
    
    # Get output filename from input path
    output_file_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(f"{method}-results", output_file_name)
    
    # Ensure output directory exists
    ensure_dir(output_path)
    
    # Update config file
    with open(config_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        yaml_data['image_path'] = image_path
        yaml_data['prompt'] = prompt
        yaml_data['output_path'] = output_path
    
    with open(config_path, 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)
    
    # Run generation
    command = f"python pnp.py --method {method} --config_path {config_path}"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running generation: {str(e)}")
        return None
    
    # Return the generated image path
    output_image = os.path.join(output_path, f"output-{prompt}.png")
    return output_image if os.path.exists(output_image) else None

def process_image(image, method, prompt):
    """Main process combining preprocessing and generation"""
    if image is None:
        return gr.Warning("Please upload an image first")
        
    # Run preprocessing
    image_path = run_preprocess(image, method)
    if image_path is None:
        return gr.Warning("Preprocessing failed")
    
    # Generate result
    result_path = generate_image(image_path, method, prompt)
    
    if result_path and os.path.exists(result_path):
        return Image.open(result_path)
    return gr.Warning("Generation failed")

# Create Gradio interface
with gr.Blocks(title="360PanT Image Translation") as demo:
    gr.Markdown("# 360PanT Image Translation")
    gr.Markdown("Upload a 360-degree panorama image and specify the translation prompt.")
    
    with gr.Row():
        with gr.Column():
            # Input components
            input_image = gr.Image(type="pil", label="Input 360° Panorama")
            method = gr.Radio(
                choices=["PnP", "360PanT"],
                value="360PanT",
                label="Method"
            )
            prompt = gr.Textbox(
                lines=2,
                placeholder="Enter your prompt here (e.g., 'a photo of castle.')",
                label="Prompt"
            )
            submit_btn = gr.Button("Generate")
        
        with gr.Column():
            # Output components
            output_image = gr.Image(label="Generated Result")
    
    # Connect components
    submit_btn.click(
        process_image,
        inputs=[input_image, method, prompt],
        outputs=output_image
    )

if __name__ == "__main__":
    # Change to the script's directory first
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    demo.launch(share=True)