package cs231n;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

import javax.imageio.ImageIO;
import javax.print.attribute.standard.Finishings;

public class Main {

	static public int[] getPixelARGB(int pixel) {
		int red = (pixel >> 16) & 0xff;
		int green = (pixel >> 8) & 0xff;
		int blue = (pixel) & 0xff;
		int[] RGB = { red, green, blue };
		return RGB;
	}

	public static void main(String[] args) throws IOException {
		File folder = new File("U:\\cs231n\\cs231n\\src\\cs231n\\Images");

		LittleImage[] trainingImages = new LittleImage[50000];
		LittleImage[] testImages = new LittleImage[10000];

		// Charger les images
		loadAllImages(folder, trainingImages, testImages);
		System.out.println("on a chargé toutes les petites images");
		
		// KNN
		int k = 5;
		int succes = 0;
		int fail = 0;
		
		
		for (LittleImage testImage : testImages) {
			int[] pixelsTestImage = testImage.getPixels();
			
			int distanceIndex = 0;
			Object[][] distances = new Object[50000][2];
			
			for (LittleImage trainingImage : trainingImages) {
				int[] pixelsTrainingImage = trainingImage.getPixels();
				
				int distance = 0;
				for (int i = 0; i < pixelsTestImage.length; i++) {
					distance += Math.abs(pixelsTestImage[i] - pixelsTrainingImage[i]);
				}
				
				// on a la distance
				distances[distanceIndex][0] = distance;
				distances[distanceIndex][1] = trainingImage.getLabel();
				distanceIndex++;
			}
			
			Arrays.sort(distances, new Comparator<Object[]>() {
			    public int compare(Object[] obj1, Object[] obj2) {
			    	Integer numOfKeys1 = (Integer)obj1[0];
			    	Integer numOfKeys2 = (Integer)obj2[0];
			        return numOfKeys1.compareTo(numOfKeys2);
			    }});

			String[] labels = new String[k];
			for (int i=0; i<k; i++) {
				labels[i] = (String) distances[i][1];
			}
			String predictedLabel = findPopular(labels);
			
			if (predictedLabel.equals(testImage.getLabel())) {
				succes++;
			}
			else {
				fail++;
			}
			
			System.out.println("succes : " + succes + "| fail : " + fail);
			
		}
		
		System.out.println("succes : " + succes);
		System.out.println("fail : " + fail);

	}

	public static String findPopular(String[] a) {

		if (a == null || a.length == 0)
			return null;

		Arrays.sort(a);

		String previous = a[0];
		String popular = a[0];
		int count = 1;
		int maxCount = 1;

		for (int i = 1; i < a.length; i++) {
			if (a[i].equals(previous))
				count++;
			else {
				if (count > maxCount) {
					popular = a[i - 1];
					maxCount = count;
				}
				previous = a[i];
				count = 1;
			}
		}

		return count > maxCount ? a[a.length - 1] : popular;
	}

	private static void loadAllImages(File folder, LittleImage[] trainingImages, LittleImage[] testImages)
			throws IOException {
		int trainingImageIndex = 0;
		int testImageIndex = 0;
		for (File file : folder.listFiles()) {
			System.out.println("Image en cours: " + file.getName());

			BufferedImage image = ImageIO.read(file);
			int w = image.getWidth();
			int h = image.getHeight();

			int[][][] pixels = new int[w][h][3];

			for (int i = 0; i < w; i++) {
				for (int j = 0; j < h; j++) {
					int pixel = image.getRGB(i, j);
					pixels[i][j] = getPixelARGB(pixel);
				}
			}

			// On charge les images de training
			LittleImage[] littleTrainingImages = loadLittleImages(file, pixels, w / 32, 0, h / 32 - 10);
			for (int i = 0; i < littleTrainingImages.length; i++) {
				trainingImages[trainingImageIndex++] = littleTrainingImages[i];
			}

			// On charge les images de test
			LittleImage[] littleTestImages = loadLittleImages(file, pixels, w / 32, h / 32 - 10, h / 32);
			for (int i = 0; i < littleTestImages.length; i++) {
				testImages[testImageIndex++] = littleTestImages[i];
			}
		}
	}

	private static LittleImage[] loadLittleImages(File file, int[][][] pixels, int nbImagesX, int startY,
			int nbImagesY) {
		LittleImage[] images = new LittleImage[nbImagesX * (nbImagesY - startY)];
		int imageIndex = 0;
		for (int a = 0; a < nbImagesX; a++) {
			for (int b = startY; b < nbImagesY; b++) {
				/// pour chaque image (a,b)

				int[] littleImage = new int[32 * 32 * 3];
				int pixelIndex = 0;
				for (int i = a * 32; i < a * 32 + 32; i++) {
					for (int j = b * 32; j < b * 32 + 32; j++) {
						// pour chaque pixel
						littleImage[pixelIndex++] = pixels[i][j][0];
						littleImage[pixelIndex++] = pixels[i][j][1];
						littleImage[pixelIndex++] = pixels[i][j][2];
					}
				}

				images[imageIndex++] = new LittleImage(littleImage,
						file.getName().substring(0, file.getName().length() - 4));
			}
		}

		return images;
	}
}
