package cs231n;

public class LittleImage {
	public static final int LITTLE_IMAGE_SIZE = 32;
	
	private int[] pixels;
	private Label label;
	
	public LittleImage(int[] pixels, Label label) {
		this.pixels = pixels;
		this.label = label;
	}

	public int[] getPixels() {
		return pixels;
	}

	public Label getLabel() {
		return label;
	}
}
