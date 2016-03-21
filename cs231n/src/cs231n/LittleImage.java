package cs231n;

public class LittleImage {
	private int[] pixels;
	private String label;
	
	public LittleImage(int[] pixels, String label) {
		this.pixels = pixels;
		this.label = label;
	}

	public int[] getPixels() {
		return pixels;
	}

	public String getLabel() {
		return label;
	}
}
