def calculate_discount(price, discount_percent):
    if discount_percent >= 20:
        discount = (price * discount_percent) / 100
        final_price = price - discount
        return final_price
    else:
        return price

price = int(input("Enter your original price: "))
discount_percent = int(input("Enter your discount percentage: "))
print(calculate_discount(price, discount_percent))
