# Assignment 1
class Event:
    def __init__(self, name, location):
        self.name = name
        self.location = location

    def advert(self):
        print(f"{self.name} coming soon")

event1 = Event("Sol Fest", "Kasarani Stadium")
event1.advert()

class Race(Event):
    def __init__(self, name, location, time):
        super().__init__(name, location)
        self.time = time
    def __str__(self):
        return f"{self.name} will take place at {self.location}"
    
    def advert(self):
        print(f"The {self.name} race set to be held at {self.location} will start at {self.time} ")

Race1 = Race("WRC", "Naivasha", "9:00AM")
Race1.advert()

# Assignment 2 

class Animal:
    def __init__(self, name):
        self.name = name
    
    def sound(self):
        print("Sound")
class Dog(Animal):
    def sound(self):
        print("Barks!!")

class Cat(Animal):
    def sound(self):
        print("Meow")


dog1 = Dog("German Shepherd")
cat1 = Cat("Melly")

for c in (dog1, cat1):
    print(c.name)
    c.sound()