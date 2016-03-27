package cs231n;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;

import javax.imageio.ImageIO;

public class KNN {
	// Constantes
	private static final int LITTLE_IMAGE_SIZE = 32;
	private static final int NB_RGB_DIMENSIONS = 3;

	/**
	 * Main()
	 * 
	 * @param args
	 * @throws IOException
	 */
	public static void knn(String[] args) throws IOException {
		long startTime;

		LittleImage[] trainingImages = new LittleImage[50000];
		LittleImage[] testImages = new LittleImage[10000];

		// On charge les images
		startTime = System.currentTimeMillis();
		loadImages(trainingImages, testImages);
		calculateAndDisplayTimeElapsed(startTime, "Images chargées");

		// On fait l'algo k-NN avec k=5
		startTime = System.currentTimeMillis();
		doKNearestNeighbours(trainingImages, testImages, 5);
		calculateAndDisplayTimeElapsed(startTime, "k-NN avec k=5 exécuté");
	}

	private static void doKNearestNeighbours(LittleImage[] trainingImages, LittleImage[] testImages, int k) {
		System.out.println("k-NN avec k=" + k + " commencé.");

		int succes = 0;
		int fail = 0;
		
		long startTime = System.currentTimeMillis();

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
					Integer numOfKeys1 = (Integer) obj1[0];
					Integer numOfKeys2 = (Integer) obj2[0];
					return numOfKeys1.compareTo(numOfKeys2);
				}
			});

			String[] labels = new String[k];
			for (int i = 0; i < k; i++) {
				labels[i] = ((Label) distances[i][1]).toString();
			}
			String predictedLabel = findPopularString(labels);

			if ((succes + fail) % 10 == 0) {
				System.out.println("-> Succès : " + succes + " | Echecs : " + fail);
			}

			if (succes + fail == 100) {
				estimateAndDisplayTimeNeeded(startTime);
			}

			if (predictedLabel.equals(testImage.getLabel())) {
				succes++;
			} else {
				fail++;
			}
		}

		int pourcentageDeReussite = (succes) / (succes + fail) * 100;
		System.out.println("k-NN avec k=" + k + " terminé.");
		System.out.println("------------------------------");
		System.out.println("Résultats : ");
		System.out.println("Succès : " + succes);
		System.out.println("Echecs : " + fail);
		System.out.println("Pourcentage de réussite : " + pourcentageDeReussite);
	}

	/**
	 * Retourne la string la plus présente dans un tableau de strings.
	 * 
	 * @param stringArray
	 *            le tableau de strings
	 * @return la string la plus présente
	 */
	public static String findPopularString(String[] stringArray) {
		if (stringArray == null || stringArray.length == 0)
			return null;
		Arrays.sort(stringArray);
		String previous = stringArray[0];
		String popular = stringArray[0];
		int count = 1;
		int maxCount = 1;
		for (int i = 1; i < stringArray.length; i++) {
			if (stringArray[i].equals(previous))
				count++;
			else {
				if (count > maxCount) {
					popular = stringArray[i - 1];
					maxCount = count;
				}
				previous = stringArray[i];
				count = 1;
			}
		}
		return count > maxCount ? stringArray[stringArray.length - 1] : popular;
	}

	/**
	 * Charge les images de training et de test.
	 * 
	 * @param trainingImages
	 * @param testImages
	 * @throws IOException
	 */
	private static void loadImages(LittleImage[] trainingImages, LittleImage[] testImages) throws IOException {
		System.out.println("Chargement des images.");

		// On spécifie le nom du dossier où trouver les images
		File imagesFolder = new File("Images");

		int trainingImageIndex = 0;
		int testImageIndex = 0;
		for (File imageFile : imagesFolder.listFiles()) {
			// On récupère le nom de la classe
			String className = imageFile.getName().substring(0, imageFile.getName().length() - 4);
			System.out.println("-> images de la classe " + className);

			BufferedImage bufferedImage = ImageIO.read(imageFile);
			int imageWidth = bufferedImage.getWidth();
			int imageHeight = bufferedImage.getHeight();

			int[][][] pixels = new int[imageWidth][imageHeight][NB_RGB_DIMENSIONS];

			for (int i = 0; i < imageWidth; i++) {
				for (int j = 0; j < imageHeight; j++) {
					int pixel = bufferedImage.getRGB(i, j);
					pixels[i][j] = pixelToRgbArray(pixel);
				}
			}

			// On charge les images de training
			LittleImage[] littleTrainingImages = loadLittleImages(className, pixels, imageWidth / LITTLE_IMAGE_SIZE, 0, imageHeight / LITTLE_IMAGE_SIZE - 10);
			for (int i = 0; i < littleTrainingImages.length; i++) {
				trainingImages[trainingImageIndex++] = littleTrainingImages[i];
			}

			// On charge les images de test
			LittleImage[] littleTestImages = loadLittleImages(className, pixels, imageWidth / LITTLE_IMAGE_SIZE, imageHeight / LITTLE_IMAGE_SIZE - 10, imageHeight / LITTLE_IMAGE_SIZE);
			for (int i = 0; i < littleTestImages.length; i++) {
				testImages[testImageIndex++] = littleTestImages[i];
			}
		}
		System.out.println("Images chargées.");
	}

	/**
	 * @param className
	 *            nom de la classe
	 * @param pixels
	 *            tableau des pixels RGB
	 * @param nbColumnsWanted
	 *            nombre de colonnes d'images voulu
	 * @param nbLinesToSkip
	 *            nombre de lignes d'images à ne pas prendre
	 * @param nbLinesWanted
	 *            nombre de lignes d'images voulu
	 * @return une liste de LittleImages (32x32)
	 */
	private static LittleImage[] loadLittleImages(String className, int[][][] pixels, int nbColumnsWanted, int nbLinesToSkip, int nbLinesWanted) {
		// on initialise une liste de littleImages
		LittleImage[] littleImages = new LittleImage[nbColumnsWanted * (nbLinesWanted - nbLinesToSkip)];

		int imageIndex = 0;
		for (int a = 0; a < nbColumnsWanted; a++) {
			for (int b = nbLinesToSkip; b < nbLinesWanted; b++) {
				/// pour chaque image (a,b), on créé une littleImage
				int[] littleImage = new int[LITTLE_IMAGE_SIZE * LITTLE_IMAGE_SIZE * NB_RGB_DIMENSIONS];

				int pixelIndex = 0;
				for (int i = a * LITTLE_IMAGE_SIZE; i < a * LITTLE_IMAGE_SIZE + LITTLE_IMAGE_SIZE; i++) {
					for (int j = b * LITTLE_IMAGE_SIZE; j < b * LITTLE_IMAGE_SIZE + LITTLE_IMAGE_SIZE; j++) {
						// on ajoute chaque pixel de l'image dans littleImage
						littleImage[pixelIndex++] = pixels[i][j][0];
						littleImage[pixelIndex++] = pixels[i][j][1];
						littleImage[pixelIndex++] = pixels[i][j][2];
					}
				}

				// on ajoute la littleImage dans la liste littleImages
				littleImages[imageIndex++] = new LittleImage(littleImage, Label.fromVal(className));
			}
		}

		// on retourne la liste des littleImages
		return littleImages;
	}

	/**
	 * Transforme la valeur d'un pixel en tableau RGB
	 * 
	 * @param pixel
	 *            valeur du pixel. Ex: -18726231621
	 * @return tableau RGB. Ex: [255, 100, 87]
	 */
	public static int[] pixelToRgbArray(int pixel) {
		int red = (pixel >> 16) & 0xff;
		int green = (pixel >> 8) & 0xff;
		int blue = (pixel) & 0xff;
		int[] RGB = { red, green, blue };
		return RGB;
	}

	/**
	 * Calcule et affiche le temps écoulé depuis un temps de départ.
	 * 
	 * @param startTime
	 *            le temps au départ
	 */
	private static void calculateAndDisplayTimeElapsed(long startTime, String message) {
		long endTime = System.currentTimeMillis();

		// Temps en secondes
		int totalRunningTime = (int) (((float) (endTime - startTime)) / 1000f);
		String unite = "secondes";

		// Temps en minutes si plus d'une minute
		if (totalRunningTime > 59) {
			totalRunningTime = (int) (totalRunningTime / 60);
			unite = "minutes";
		}

		System.out.println("=================");
		System.out.println(message + " en " + totalRunningTime + " " + unite + ".");
		System.out.println("=================");
	}

	/**
	 * Calcule et affiche le temps d'éxecution estimé.
	 * 
	 * @param startTime
	 *            le temps au départ
	 */
	private static void estimateAndDisplayTimeNeeded(long startTime) {
		long endTime = System.currentTimeMillis();

		// Temps en secondes
		int totalRunningTime = (int) (((float) (endTime - startTime)) / 1000f);
		totalRunningTime = totalRunningTime * 100;
		String unite = "secondes";

		// Temps en minutes si plus d'une minute
		if (totalRunningTime > 59) {
			totalRunningTime = (int) (totalRunningTime / 60);
			unite = "minutes";
		}

		System.out.println("=================");
		System.out.println("-> Le programme devrait se terminer en " + totalRunningTime + " " + unite + ".");
		System.out.println("=================");
	}
}
