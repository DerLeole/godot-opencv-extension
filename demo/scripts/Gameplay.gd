extends Node

var camera : CVCamera = CVCamera.new();
@export_group("Canvas")
@export var camera_canvas : Sprite2D;
@export var overlay_canvas : Sprite2D;
@export_group("Input")
@export var threshold_slider : Range;
@export var mode_button : OptionButton;
@export var overlay_button : CheckButton;
@export_group("Label")
@export var rectangle_label : Label;
var texture : ImageTexture;
var overlay_texture : ImageTexture;

func _ready():
    camera.open(0);
    camera.flip(false, false);
    texture = ImageTexture.new();
    overlay_texture = ImageTexture.new();
    threshold_slider.value_changed.connect(_on_threshold_changed);

func _process(delta):
    match mode_button.selected:
        0: # Color
            texture.set_image(camera.get_image());
            camera.get_threshold_image();
        1: # Grey
            texture.set_image(camera.get_gray_image());
            camera.get_threshold_image();
        2: # Threshold
            texture.set_image(camera.get_threshold_image());
            
    rectangle_label.text = str(camera.find_rectangles(overlay_button.button_pressed));
    
    if overlay_button.button_pressed:
        overlay_texture.set_image(camera.get_overlay_image());
    overlay_canvas.visible = overlay_button.button_pressed;
    
    camera_canvas.texture = texture;
    overlay_canvas.texture = overlay_texture;
    
    
func _on_threshold_changed(value : float):
    camera.set_threshold(value);
