"""
Northern Arizona University
Fall-2020

Course: INF638
Author: Md Nazmul Hossain (mh2752@nau.edu)

"""

import numpy as np
import random
import math


# ------------ Global Varibles ---------------------

# Field detail:
m = 4 # 2^m
g_field_size = pow(2,4)

# Irreducible polynomial:
P_x = np.poly1d([1,0,0,1,1]) # x^4+x+1


# The dictionaries for holding all the generator g of the field in different format:
all_g_in_poly1d_format = {}
all_g_in_digital_string_format = {}
all_field_elements = {}
g_power_list = {}


# -----------------------------------------------------



def generate_all_g_in_poly1d_format(m):

	gs_in_poly1d_format = {}

	for i in range(0,g_field_size):

		key = "g"+str(i)

		list_for_poly1d = [1]

		j = 0

		while j<i:

			# Padding necessary 0s to the list:
			list_for_poly1d.append(0)
			j += 1

		poly1d_point = np.poly1d(list_for_poly1d)


		# Expressing the point in terms of the irreducible polynomial:
		quotient,remainder = np.polydiv(poly1d_point,P_x)
		remainder_coefficients = (abs(remainder.c))%2

		gs_in_poly1d_format[key] = remainder_coefficients

		g_power_list[key] = i # E.g => g^0 = 0, g^3 = 3 etc.

	zeropoly1d = np.poly1d([0])
	q,r = np.polydiv(zeropoly1d,P_x)
	r_coeffs = (abs(r.c))%2

	gs_in_poly1d_format["0"] = r_coeffs


	return gs_in_poly1d_format


# ----------------------------------------------------------

def generate_all_g_in_digital_format(gs_poly1d_dictionary):

	gs_in_digital_string_format = {}

	gs_list = list(gs_poly1d_dictionary.keys())

	for g in gs_list:

		g_coefficient = list(gs_poly1d_dictionary[g])

		# Equalizing the length by padding necessary 0s at the front:
		while(len(g_coefficient)<4):
			g_coefficient.insert(0,0.0)

		gs_in_digital_string_format[g] = "".join(str(int(item)) for item in g_coefficient)

		gs_in_digital_string_format["0"] = "0000"

	return gs_in_digital_string_format


# ----------------------------------------------------------


def add_two_g(g_a,g_b):

	result = ""		

	g_a_digital_string = all_g_in_digital_string_format[g_a]
	g_b_digital_string = all_g_in_digital_string_format[g_b]

	for i in range(0,4):

		if g_a_digital_string[i] != g_b_digital_string[i]:

			result += str(1)

		else:

			result += str(0)

		
	return find_g_by_digital_string_value(result)
	



def find_g_by_digital_string_value(g_digital_string_value):

	g = list(all_g_in_digital_string_format.keys())[list(all_g_in_digital_string_format.values()).index(g_digital_string_value)]

	return g



def find_g_inverse(g_name):

	if g_name=="0":

		return "Infinity"

	g_pow = g_power_list[g_name]

	inverse_power = (-1*(g_pow))%(g_field_size-1)

	inversed_g = list(g_power_list.keys())[list(g_power_list.values()).index(inverse_power)]

	return inversed_g


def mul_two_g(g_a_name,g_b_name):

	if (g_a_name=="0" or g_b_name=="0"):
		
		return "0"

	g_a_power = g_power_list[g_a_name]
	g_b_power = g_power_list[g_b_name]

	prod_power = (g_a_power+g_b_power) % (g_field_size-1)

	prod_g = list(g_power_list.keys())[list(g_power_list.values()).index(prod_power)]

	return prod_g


def square_a_g(g_a_name):

	if (g_a_name=="0"):
		
		return "0"

	g_a_power = g_power_list[g_a_name]
	
	sq_power = (g_a_power+g_a_power) % (g_field_size-1)

	squared_g = list(g_power_list.keys())[list(g_power_list.values()).index(sq_power)]

	return squared_g



def validate_expression(g_list):

	unique_gs = set(g_list)

	unique_g_freq = {}

	for g in unique_gs:

		freq = g_list.count(g)
		unique_g_freq[g] = freq


	# If an item occurs even times, remove completely. Else, keep only one occurence:

	for g in unique_gs:

		if unique_g_freq[g]%2==0:

			while g in g_list:
				g_list.remove(g)
		else:

			while g in g_list:
				g_list.remove(g)

			g_list.append(g)

	return g_list



def get_ecc_key(point,n):

	keys = list(all_field_elements.keys())
	vals = list(all_field_elements.values())
	
	point_position = keys[vals.index(point)]

	final_position = (point_position*n)%g_field_size
	temp_list = all_field_elements[final_position]
	return temp_list[0],temp_list[1]


def generator_for_the_field(primX,primY,curve_param):


	counter = 0

	while counter<g_field_size:

		if counter==0:
			
			all_field_elements[counter] = ["0","0"]
		
		elif counter == 1:

			all_field_elements[counter] = [primX,primY]

		elif counter == 2:

			x3, y3 = ecc_point_doubling(primX,primY,curve_param)
			all_field_elements[counter] = [x3,y3]
		
		else:
			last_element = all_field_elements[counter-1]
			#print(last_element)
			x1 = last_element[0]
			y1 = last_element[1]
			#print("x1,y1 = ",x1,y1)
			x3,y3 = ecc_point_addition(x1,y1,primX,primY,curve_param)
			all_field_elements[counter] = [x3,y3]

		counter += 1

	return





########################################## POINT ADDITION IMPLEMENTATION ############################################################



def ecc_point_addition(g_x1,g_y1,g_x2,g_y2,g_a):

	a1 = add_two_g(g_y2,g_y1)
	b1 = add_two_g(g_x2,g_x1)
	
	b1_inv = find_g_inverse(b1)

	if b1_inv == "Infinity": # Slope shall be undefined at the point of Infinity for the neutral point

		#print("----- Result of Extended Field ECC Point Addition ------")
		#print("(%s,%s) + (%s,%s) = %s "%(g_x1,g_y1,g_x2,g_y2,"0"))
		return "0","0"

	slope = mul_two_g(a1,b1_inv)

	
	# Computing x3:
	slope_squared = square_a_g(slope)

	x3_expression = [slope_squared,slope,g_x1,g_x2,g_a]
	x3_expression = validate_expression(x3_expression)
	
	while(len(x3_expression)>1):
		
		item1 = x3_expression.pop(0)
		item2 = x3_expression.pop(0)

		sum_result = add_two_g(item1,item2)

		if sum_result!=None:
			x3_expression.append(sum_result)

		x3_expression = validate_expression(x3_expression)

	x3_value = "0"

	if (len(x3_expression)!=0):
		x3_value = x3_expression[0]

	

	# Calculating y3 value:

	sum1 = add_two_g(g_x1,x3_value)


	if sum1==None:

		print("Error: x1 and x3 same Detected.")
		return

	prod1 = mul_two_g(slope,sum1)

	
	y3_expression = [prod1,x3_value,g_y1]
	
	y3_expression = validate_expression(y3_expression)


	while(len(y3_expression)>1):
		
		item1 = y3_expression.pop(0)
		item2 = y3_expression.pop(0)

		sum_result = add_two_g(item1,item2)
		
		if sum_result!=None:
			y3_expression.append(sum_result)

		
		y3_expression = validate_expression(y3_expression)

	y3_value = "0"

	if (len(y3_expression)!=0):
		y3_value = y3_expression[0]


	#print("----- Result of Extended Field ECC Point Addition ------")
	#print("(%s,%s) + (%s,%s) = (%s,%s)"%(g_x1,g_y1,g_x2,g_y2,x3_value,y3_value))
	return x3_value,y3_value


######################################### POINT DOUBLING IMPLEMENTATION #############################################################


def ecc_point_doubling(g_x1,g_y1,g_a):
	
	
	x1_inv = find_g_inverse(g_x1)

	if x1_inv == "Infinity":

		#print("----- Result of Extended Field ECC Point Doubling ------")
		#print("(%s,%s) + (%s,%s) = %s"%(g_x1,g_y1,g_x1,g_y1,"0"))

		return "0","0"


	y1_mul_x1_inv = mul_two_g(g_y1,x1_inv)

	
	slope = ""

	if (x1_inv!="0"):

		slope_expression = [g_x1,y1_mul_x1_inv]
		slope_expression = validate_expression(slope_expression)

		if len(slope_expression)==0:

			print("Error: Could Not Calculate Slope.")
			return

		while(len(slope_expression)>1):
			
			item1 = slope_expression.pop(0)
			item2 = slope_expression.pop(0)

			sum_result = add_two_g(item1,item2)

			if sum_result!=None:
				slope_expression.append(sum_result)

			slope_expression = validate_expression(slope_expression)


		if (len(slope_expression)==0):
			print("Error: Could Not Calculate Slope2.")
			return

		slope = slope_expression[0]


	else:
		slope = "0"	


	# Computing x3:
	slope_squared = square_a_g(slope)

	x3_expression = [slope_squared,slope,g_a]
	x3_expression = validate_expression(x3_expression)
	
	while(len(x3_expression)>1):
		
		item1 = x3_expression.pop(0)
		item2 = x3_expression.pop(0)

		sum_result = add_two_g(item1,item2)

		if sum_result!=None:
			x3_expression.append(sum_result)

		x3_expression = validate_expression(x3_expression)


	x3_value = "0"

	if (len(x3_expression)>0):
		x3_value = x3_expression[0]

	

	# Calculating y3 value:

	if x3_value != "0":
		
		sum1 = add_two_g(g_x1,x3_value)

		if sum1==None:

			print("Error: x1 and x3 same Detected.")
			return
		

		prod1 = mul_two_g(slope,sum1)
		y3_expression = [prod1,x3_value,g_y1]
		y3_expression = validate_expression(y3_expression)


		while(len(y3_expression)>1):
		
			item1 = y3_expression.pop(0)
			item2 = y3_expression.pop(0)	

			sum_result = add_two_g(item1,item2)

			if sum_result!=None:
				y3_expression.append(sum_result)

			
			y3_expression = validate_expression(y3_expression)
			

		y3_value = y3_expression[0]


		#print("----- Result of Extended Field ECC Point Doubling ------")
		#print("(%s,%s) + (%s,%s) = (%s,%s)"%(g_x1,g_y1,g_x1,g_y1,x3_value,y3_value))
		return x3_value,y3_value

	else:

		prod1 = mul_two_g(slope,g_x1)
		y3_expression = [prod1,g_y1]
		y3_expression = validate_expression(y3_expression)


		while(len(y3_expression)>1):
		
			item1 = y3_expression.pop(0)
			item2 = y3_expression.pop(0)	

			sum_result = add_two_g(item1,item2)

			if sum_result!=None:
				y3_expression.append(sum_result)

			
			y3_expression = validate_expression(y3_expression)
			

		y3_value = "0"

		if (len(y3_value)!=0):
			y3_value = y3_expression[0]


		#print("----- Result of Extended Field ECC Point Doubling ------")
		#print("(%s,%s) + (%s,%s) = (%s,%s)"%(g_x1,g_y1,g_x1,g_y1,x3_value,y3_value))

		return x3_value,y3_value


######################################### EC Diffie-Hellman Key Exchange ######################



def diffie_hellman(primitive_element,curve_param):

	
	prim = primitive_element.split(",")
	generator_for_the_field(prim[0],prim[1],curve_param)

	
	# Calculation for Alice:
	a = random.randrange(2,g_field_size-1) # Private key of Alice
	x_A,y_A = get_ecc_key(prim,a) # Public key componenets for Alice's public key
	public_Alice = [x_A,y_A]


	# Calculation for Bob:
	b = random.randrange(2,g_field_size-1)
	x_B,y_B = get_ecc_key(prim,b)
	public_Bob = [x_B,y_B]


	# Calculating Secret Key:
	Secret_Alice_x,Secret_Alice_y = get_ecc_key(public_Bob,a)
	Secret_Bob_x,Secret_Bob_y = get_ecc_key(public_Alice,b)

	Secret_key_Alice = [Secret_Alice_x,Secret_Alice_y]
	Secret_key_Bob = [Secret_Bob_x,Secret_Bob_y]

	
	print("Private Key for Alice = ",a)
	print("Public Key for Alice = ",public_Alice)

	print("Private Key for Bob = ",b)
	print("Public Key for Bob = ",public_Bob)

	print("Secret Key for Alice = ", Secret_key_Alice)
	print("Secret Key for Bob = ", Secret_key_Bob)

	

	return

################################################ EC (Extended Galois Field) DSA #########################################################

# Returns the inverse for a^(-1) mod m
def get_inverse(a,m):

	counter = 1

	while counter<m:

		if (a*counter)%m == 1:

			return counter

		counter += 1

	return None


# ------------------------------------------------------------------------------------------------------

def ec_extended_dsa():

	
	# Read the primmitive element (as g):
	A = input("Enter your primitive element (E.g: gX,gY): ")
	prim_as_list = A.split(",")	


	# Read the curve paprameter (as g)L
	curve_parameter_a = input("Enter curve parameter a (E.g: g3 or g4): ")
	generator_for_the_field(prim_as_list[0],prim_as_list[1],curve_parameter_a)

	
	# Read the message hash to be signed:
	h_x = int(input("Enter the Message Hash h(X): "))
	q = int(input("Enter the Parameter q (num of elements in the cyclic group ):"))


	# Read the secret key:
	d = int(input("Enter Your Secret Key d (d<q):"))
	
	# Computing B:
	g_Bx,g_By = get_ecc_key(prim_as_list,d)


	#-----------------------------------------------------------------------------

	s = '' # For holding the signature
	
	gcd_condition_satisfied = False

	while not(gcd_condition_satisfied):

		# obtaining ephemeral key k_e which satisfies
		# the conditions 0< k_e < q and gcd(k_e,q) == 1	
		k_e = random.randint(1,q-1)

		while math.gcd(k_e,q) != 1:

			k_e = random.randint(1,q-1)


		#-----------------------------------------------------------------------------

		# Obtaining point R where R = k_e * primitive_element
		R_gx,R_gy = get_ecc_key(prim_as_list,k_e)

		# Calculating r in decimal form from R_gx:
		r = all_g_in_digital_string_format[R_gx]
		r_as_int = int(r,2)	
		
		print("\nEphemeral Key, k_e = ",k_e," and r = ",r_as_int)


		#-----------------------------------------------------------------------------


		# Calculate the signature s = ( h(x)+ d*r ) * inv(k_e) mod q:
		inv_k_e = get_inverse(k_e,q)

		h_x_plus_dr_inv_k_e = ( h_x + (d*r_as_int)) * inv_k_e

		s = h_x_plus_dr_inv_k_e % q

		gcd_condition_satisfied = math.gcd(s,q) == 1

	# ----------------------------------------------------------------------------------

	print("\n------- EC (Extneded) DSA Results ----------")
	print("Message hash h(X) = ",h_x)
	print("r = ",r_as_int)
	print("inv(k_e) = ",inv_k_e)
	print("Signature, s = ",s)

	print("\n\n")

	print("-------- Verifying the DSA Results ---------")

	w = get_inverse(s,q)%q
	u1 = w * h_x
	u2 = (w*r_as_int)%q

	# Obtaining u1 * A:
	gX_u1A,gY_u1A = get_ecc_key(prim_as_list,u1) 

	
	# Obtaining u2 * B
	B_coord_list = [g_Bx,g_By]	
	gX_u2B,gY_u2B = get_ecc_key(B_coord_list,u2)


	# Obtaining P = u1*A + u2*B:
	gX_P,gY_P = ecc_point_addition(gX_u1A,gY_u1A,gX_u2B,gY_u2B,curve_parameter_a)


	# Verifying:

	gX_P_decimal_form = int(all_g_in_digital_string_format[gX_P],2)

	print("r = ",r_as_int," and gX_P = ",gX_P_decimal_form)

	if r_as_int == gX_P_decimal_form:

		print("Signature is Authentic.")
	
	else:

		print("Invalid Signature.")


	return



################################### EC (Extended) DSA End ######################################





################################# Initializing Global Variables ###############################

all_g_in_poly1d_format = generate_all_g_in_poly1d_format(m)
all_g_in_digital_string_format = generate_all_g_in_digital_format(all_g_in_poly1d_format) 

################################################################################################

user_choice = input("Enter Your Choice: \n1 For Point Addition\n2 For Point Doubling\n3 For EC Diffie-Hellman Demonstration\n4 For EC-DSA\n")

if user_choice == "1":

	# User input to get (x_1,y_1), (x_2,y_2) in gX (E.g: g2 0r g3) format:
	x_1 = input("Enter x_1 in g form (e.g: g3):")
	y_1 = input("Enter y_1 in g form (e.g: g4):")

	x_2 = input("Enter x_2 in g form (e.g: g5):")
	y_2 = input("Enter y_2 in g form (e.g: g6):")


	coeffs_curve_a = input("Enter the Co-efficient a for the Elliptic Curve Equation:\n"+
	" y^2 + xy = x^3 + a * (x^2) + b in g form (e.g: g9):")

	x3,y3 = ecc_point_addition(x_1,y_1,x_2,y_2,coeffs_curve_a)
	print("----- Result of Extended Field ECC Point Addition ------")
	print("(%s,%s) + (%s,%s) = (%s,%s)"%(x_1,y_1,x_2,y_2,x3,y3))

elif user_choice == "2":

	x_1 = input("Enter x_1 in g form (e.g: g3):")
	y_1 = input("Enter y_1 in g form (e.g: g4):")


	coeffs_curve_a = input("Enter the Co-efficient a for the Elliptic Curve Equation:\n"+
	" y^2 + xy = x^3 + a * (x^2) + b in g form (e.g: g9):")

	x3,y3 = ecc_point_doubling(x_1,y_1,coeffs_curve_a)
	print("----- Result of Extended Field ECC Point Doubling ------")
	print("(%s,%s) + (%s,%s) = (%s,%s)"%(x_1,y_1,x_1,y_1,x3,y3))

elif user_choice == "3":

	# Call EC Diffie-Hellman Key Exchange Function
	primitive_element = input("Enter your primitive element (E.g: gX,gY): ")
	curve_parameter_a = input("Enter curve parameter a (E.g: g3 or g4: ")
	diffie_hellman(primitive_element,curve_parameter_a)

else:
	ec_extended_dsa()

###################################################### END OF PROGRAM ##########################################################