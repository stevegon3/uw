import math

def calculate_circle_area(radius):
    """
    Calculate the area of a circle given its radius.
    
    Args:
        radius: The radius of the circle (must be a positive number)
        
    Returns:
        The area of the circle
    """
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return math.pi * (radius ** 2)

# Example usage
if __name__ == "__main__":
    try:
        radius = float(input("Enter the radius of the circle: "))
        area = calculate_circle_area(radius)
        print(f"The area of a circle with radius {radius} is: {area:.2f}")
    except ValueError as e:
        print(f"Error: {e}")
