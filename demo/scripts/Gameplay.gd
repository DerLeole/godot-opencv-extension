extends Node

var camera : CVCamera = CVCamera.new();
@export var camera_canvas : Sprite2D;
@export var threshold_slider : Range;
var texture : ImageTexture;

func _init():
    camera.open(0);
    camera.flip(false, false);
    texture = ImageTexture.new();

func _process(delta):
    texture.set_image(camera.get_image());
    camera_canvas.texture = texture;
