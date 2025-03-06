using Flux, Images, ImageIO

# Load and preprocess the image
function load_preprocessed_image(path)
    img = load(path)                         # Load the image
    img_resized = imresize(img, (224, 224))  # Resize to 224x224
    img_tensor = Float32.(channelview(img_resized))  # Convert to Float32 tensor

    println("Resized image size: ", size(img_tensor))  # Debugging step

    img_tensor = reshape(img_tensor, (224, 224, 1, 1))  # Ensure (H, W, C, Batch)
    return img_tensor
end

# Define the CNN model (WITHOUT the Dense layer)
feature_extractor = Chain(
    Conv((3,3), 1=>8, relu),  # Conv Layer (input: 1, output: 8)
    MaxPool((2,2)),           # Reduces size to (112,112)
    Conv((3,3), 8=>16, relu), # Conv Layer (input: 8, output: 16)
    MaxPool((2,2)),           # Reduces size to (56,56)
    Flux.flatten              # Flatten layer
)

# Compute correct input size for Dense layer
dummy_input = rand(Float32, 224, 224, 1, 1)  # Random tensor simulating an image
flattened_output = feature_extractor(dummy_input)
dense_input_size = size(flattened_output, 1)  # Get correct flattened size
println("Computed Dense layer input size: ", dense_input_size)

# Now define the FULL model with correct Dense input size
model = Chain(
    feature_extractor,
    Dense(dense_input_size, 10, relu)  # Use computed input size
)

# Load and preprocess the actual image
img_tensor = load_preprocessed_image("C:/Users/aakas/OneDrive/Desktop/New folder/Image Preprocessing/preprocessed_images/processed_astronaut_rides_car_1.png")

# Check shape before forward pass
println("Tensor shape before model: ", size(img_tensor))

# Forward pass
output = model(img_tensor)
println("Model Output: ", output)




# Model Output: Float32[0.003821756; 0.06374647; 0.052354507; 0.03351292; 0.035395026; 0.04086785; 0.0; 0.092709824; 0.030221522; 0.0;;]