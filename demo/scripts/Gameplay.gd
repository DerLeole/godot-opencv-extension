extends Node

var camera : CVCamera = CVCamera.new();
@onready
var camera_canvas : Sprite2D = $CameraCanvas;
var texture : ImageTexture;

func _init():
    camera.open(0);
    texture = ImageTexture.new();

func _process(delta):
    texture.set_image(camera.get_image());
    camera_canvas.texture = texture;
