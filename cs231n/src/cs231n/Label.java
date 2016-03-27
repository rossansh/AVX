package cs231n;

public enum Label {

	AIRPLANE,
	AUTOMOBILE,
	BIRD,
	CAT,
	DEER,
	DOG,
	FROG,
	HORSE,
	SHIP,
	TRUCK;
		
	public static Label fromVal(String className) {
		return Label.valueOf(className.toUpperCase());
	}
}