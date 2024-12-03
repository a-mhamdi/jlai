using Markdown 
using Images

md"Load an image and predict the digit"
img_path = "1.png"
img = load(img_path);
md"Resize the image to 28x28"
resized_img = imresize(Gray.(img), (28, 28));
md"Add two extra dimensions ...x1x1"
reshaped_img = reshape(resized_img, 28, 28, 1, 1);

using Flux, BSON
BSON.@load "cnn.bson" model;

model(reshaped_img) |> (x -> Flux.onecold(x) .- 1)  # Predict the digit
