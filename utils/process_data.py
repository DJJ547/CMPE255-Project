
# to calculate BMI, we need to convert height in feet and inchs to meters, and weight in pounds to kilograms
def calculate_bmi(weight_lb, height_ft, height_inch):
    weight_lb = float(weight_lb)
    height_ft = float(height_ft)
    height_inch = float(height_inch)
    weight_in_kg = weight_lb * 0.45359237
    height_m_1 = height_ft * 0.3048
    height_m_2 = height_inch * 0.0254
    height_m = height_m_1 + height_m_2
    bmi = weight_in_kg / height_m
    return str(int(round(bmi, 0)))

# heavy drinker: adult men having more than 14 drinks per week and adult women having more than 7 drinks per week
def determine_heavy_drinker(gender, num_alcohol):
    num_alcohol = int(num_alcohol)
    if (gender == 'male' and num_alcohol > 14) or (gender == 'female' and num_alcohol > 7):
        return 1
    else:
        return 0
