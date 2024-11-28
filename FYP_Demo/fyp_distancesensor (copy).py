# Student Name: 	Ciaran Carroll
# Student Id: 		13113259
#
# Python script 
#
# The trigger sends out a burst and the echo takes it in using the known constant that is the 
# speed of sound we can calculate how much distance was travelled based on how long it took 
# for that sound wave to go out and come back

# Import the GPIO and the time library 
import RPi.GPIO as GPIO
import time

# GPIO Mode (BOARD / BCM)
GPIO.setmode(GPIO.BCM)

# Set GPIO Pins
GPIO_TRIGGER = 23
GPIO_ECHO = 24

print('Distance Measurement In Progress')

# Set the GPIO direction (IN / OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

# Ensure the Trigger pin set to low, and give the sensor a second to settle
GPIO.output(GPIO_TRIGGER, False)
print('Waiting For Sensor To Settle')
time.sleep(2)

# Set Trigger to HIGH
GPIO.output(GPIO_TRIGGER, True)

# Set Trigger after 0.01ms to LOW 
time.sleep(0.0001)
GPIO.output(GPIO_TRIGGER, False)

StartTime = time.time()
StopTime = time.time()

# Save StartTime
while GPIO.input(GPIO_ECHO) == 0:
	StartTime = time.time()
	
# Save time of arrival	
while GPIO.input(GPIO_ECHO) == 1:
	StopTime = time.time()

# Calculate Time difference between start and arrival
TimeElapsed = StopTime - StartTime
	
# Calculation of the Dinstance in cms
Distance = TimeElapsed / 0.000058

# Prints the Distance
print('Distance: {} cm'.format(Distance))

# Clean the GPIO pins to ensure that all the inputs/outputs are reset
GPIO.cleanup()