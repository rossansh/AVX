package cs231n;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import javax.imageio.ImageIO;

public class Main {
	private static final int NB_RGB_DIMENSIONS = 3;

	/**
	 * Main()
	 */
	public static void main(String[] args) throws IOException {
		LittleImage[] trainingImages = new LittleImage[50000];
		LittleImage[] testImages = new LittleImage[10000];

		// On charge les images
		loadImages(trainingImages, testImages);

		// Entrainement du W
		long startTime = System.currentTimeMillis();
		double[][] W;
		
		// Paramètres du programme
		int startingW = 1;
		int numberOfTrainingsToDo = 1000;
		
		File startingWFile = new File("saving/W_" + startingW + ".ser");
		
		if (startingWFile.exists()) {
			W = loadWfromFile(startingWFile);
			System.out.println("W_" + startingW + " chargé à partir du fichier");
			System.out.println("------------------------------");
		}
		else {
			W = Matrix.random(10, LittleImage.LITTLE_IMAGE_SIZE * LittleImage.LITTLE_IMAGE_SIZE * NB_RGB_DIMENSIONS + 1);
			saveWtoFile(W, new File("saving/W_1.ser"));
			System.out.println("W_" + startingW + " initialisé au hasard.");
			System.out.println("------------------------------");
			
			for (int i = 2; i <= startingW; ++i) {
				W = trainAndTestW(trainingImages, testImages, W, i);
				calculateAndDisplayTimeElapsed(startTime, "W_" + i + " entraîné et testé");
				startTime = System.currentTimeMillis();
			}
		}
		
		for (int i = startingW + 1; i <= startingW + numberOfTrainingsToDo; ++i) {
			W = trainAndTestW(trainingImages, testImages, W, i);
			calculateAndDisplayTimeElapsed(startTime, "W_" + i + " entraîné et testé");
			startTime = System.currentTimeMillis();
		}
	}

	private static double[][] trainAndTestW(LittleImage[] trainingImages, LittleImage[] testImages, double[][] W, int i) throws IOException {
		System.out.println("W - Entrainement n°" + i + " commencé.");
		W = trainW(trainingImages, W);
		saveWtoFile(W, new File("saving/W_" + i + ".ser"));
		testW(testImages, W);
		System.out.println("W - Entrainement n°" + i + " terminé.");
		return W;
	}

	private static void testW(LittleImage[] testImages, double[][] W) {
		// Test
		int success = 0;
		int fail = 0;

		for (LittleImage testImage : testImages) {
			double[] scores = Matrix.multiply(W, testImage.getPixels());
			int indexMax = 0;
			double max = scores[indexMax];

			for (int i = 1; i < scores.length; i++) {
				if (scores[i] > max) {
					max = scores[i];
					indexMax = i;
				}
			}

			Label predictedLabel = Label.values()[indexMax];
			if (predictedLabel == testImage.getLabel()) {
				success++;
			} else {
				fail++;
			}
		}

		float pourcentageDeReussite = (float)(success) / (float)(success + fail) * 100;
		System.out.println("------------------------------");
		System.out.println("Résultats : ");
		System.out.println("Succès : " + success);
		System.out.println("Echecs : " + fail);
		System.out.println("Pourcentage de réussite : " + pourcentageDeReussite);
		System.out.println("------------------------------");
	}

	private static void saveWtoFile(double[][] W, File wfile) throws IOException {
		ObjectOutputStream oos = null;
		try {
			FileOutputStream fout = new FileOutputStream(wfile, true);
			oos = new ObjectOutputStream(fout);
			oos.writeObject(W);
		} catch (Exception ex) {
			ex.printStackTrace();
		} finally {
			if (oos != null) {
				oos.close();
			}
		}
	}
	
	private static double[][] loadWfromFile(File wfile) throws IOException {
		ObjectInputStream objectinputstream = null;
		try {
		    FileInputStream streamIn = new FileInputStream(wfile);
		    objectinputstream = new ObjectInputStream(streamIn);
		    double[][] W = (double[][]) objectinputstream.readObject();
		    return W;
		} catch (Exception e) {
		    e.printStackTrace();
		} finally {
		    if(objectinputstream != null){
		        objectinputstream .close();
		    } 
		}
		return null;
	}

	private static double[][] trainW(LittleImage[] trainingImages, double[][] W) {
		double[] Li = new double[50000];
		int indexLi = 0;

		for (LittleImage trainingImage : trainingImages) {
			double[] scores = Matrix.multiply(W, trainingImage.getPixels());
			Li[indexLi++] = softmax(scores, trainingImage.getLabel());
			double currentLoss = loss(W, Li, indexLi);

			double step = 0.0001;
			double[][] gradient = new double[10][LittleImage.LITTLE_IMAGE_SIZE * LittleImage.LITTLE_IMAGE_SIZE
					* NB_RGB_DIMENSIONS + 1];
			for (int i = 0; i < scores.length; i++) {
				for (int j = 0; j < scores.length; j++) {
					W[i][j] += step;
					double loss = loss(W, Li, indexLi);
					double deriveePartielle = (loss - currentLoss) / step;
					gradient[i][j] = deriveePartielle;
					W[i][j] -= step;
				}
			}

			W = Matrix.add(W, Matrix.multiply(gradient, -step));
		}
		return W;
	}

	public static double softmax(double[] scores, Label label) {
		double goodLabelScore = scores[label.ordinal()];
		double allLabelsScores = 0;
		for (int i = 0; i < scores.length; i++) {
			allLabelsScores += scores[i];
		}
		double Li = -Math.log(Math.exp(goodLabelScore) / Math.exp(allLabelsScores));
		return Li;
	}

	public static double loss(double[][] W, double[] Li, int indexLi) {
		int lambda = 2;
		double sumOfLi = 0;

		for (int i = 0; i < indexLi; i++) {
			sumOfLi += Li[i];
		}

		double R = 0;
		for (int i = 0; i < W.length; i++) {
			for (int j = 0; j < W.length; j++) {
				R += Math.pow(W[i][j], 2);
			}
		}

		double L = (1 / 50000) * sumOfLi + lambda * R;
		return L;
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
			LittleImage[] littleTrainingImages = loadLittleImages(className, pixels,
					imageWidth / LittleImage.LITTLE_IMAGE_SIZE, 0, imageHeight / LittleImage.LITTLE_IMAGE_SIZE - 10);
			for (int i = 0; i < littleTrainingImages.length; i++) {
				trainingImages[trainingImageIndex++] = littleTrainingImages[i];
			}

			// On charge les images de test
			LittleImage[] littleTestImages = loadLittleImages(className, pixels,
					imageWidth / LittleImage.LITTLE_IMAGE_SIZE, imageHeight / LittleImage.LITTLE_IMAGE_SIZE - 10,
					imageHeight / LittleImage.LITTLE_IMAGE_SIZE);
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
	private static LittleImage[] loadLittleImages(String className, int[][][] pixels, int nbColumnsWanted,
			int nbLinesToSkip, int nbLinesWanted) {
		// on initialise une liste de littleImages
		LittleImage[] littleImages = new LittleImage[nbColumnsWanted * (nbLinesWanted - nbLinesToSkip)];

		int imageIndex = 0;
		for (int a = 0; a < nbColumnsWanted; a++) {
			for (int b = nbLinesToSkip; b < nbLinesWanted; b++) {
				// / pour chaque image (a,b), on créé une littleImage
				int[] littleImage = new int[LittleImage.LITTLE_IMAGE_SIZE * LittleImage.LITTLE_IMAGE_SIZE
						* NB_RGB_DIMENSIONS + 1];

				int pixelIndex = 0;
				for (int i = a * LittleImage.LITTLE_IMAGE_SIZE; i < a * LittleImage.LITTLE_IMAGE_SIZE
						+ LittleImage.LITTLE_IMAGE_SIZE; i++) {
					for (int j = b * LittleImage.LITTLE_IMAGE_SIZE; j < b * LittleImage.LITTLE_IMAGE_SIZE
							+ LittleImage.LITTLE_IMAGE_SIZE; j++) {
						// on ajoute chaque pixel de l'image dans littleImage
						littleImage[pixelIndex++] = pixels[i][j][0];
						littleImage[pixelIndex++] = pixels[i][j][1];
						littleImage[pixelIndex++] = pixels[i][j][2];
					}
				}

				littleImage[pixelIndex] = 1;

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
		int seconds = (int) (((float) (endTime - startTime)) / 1000f);

		System.out.println("=================");
		if (seconds > 59) {
			int minutes = (int) (seconds / 60);
			seconds = (int) (seconds % 60);
			System.out.println(message + " en " + minutes + " minutes, " + seconds + " secondes.");
		}
		else {
			System.out.println(message + " en " + seconds + " secondes.");
		}
		System.out.println("=================");

	}

}