#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample data: Vehicles with their types, models, colors, cc, prices, and manufacture years
motorcycle_data = [
    {"type": "Motorcycle", "model": "Ninja", "color": "Red", "cc": 300, "price": 12000, "year": 2020},
    {"type": "Motorcycle", "model": "Superpower", "color": "Black", "cc": 70, "price": 15000, "year": 2021},
    {"type": "Motorcycle", "model": "Honda", "color": "Black", "cc": 110, "price": 17000, "year": 2022},
    {"type": "Motorcycle", "model": "Suzuki", "color": "Blue", "cc": 250, "price": 20000, "year": 2023},
    {"type": "Motorcycle", "model": "Superpower", "color": "White", "cc": 70, "price": 18000, "year": 2001},
    {"type": "Motorcycle", "model": "H2r", "color": "Black", "cc": 400, "price": 25000, "year": 2022},
    {"type": "Motorcycle", "model": "Unionstar", "color": "Red", "cc": 350, "price": 22000, "year": 2018},
    {"type": "Motorcycle", "model": "Crown", "color": "Black", "cc": 300, "price": 18000, "year": 2021},
    {"type": "Motorcycle", "model": "Ninja", "color": "Grey", "cc": 500, "price": 30000, "year": 2003},
    {"type": "Motorcycle", "model": "Superpower", "color": "Brown", "cc": 450, "price": 58000, "year": 2009},
    {"type": "Motorcycle", "model": "Superpower", "color": "White", "cc": 70, "price": 78200, "year": 2001},
    {"type": "Motorcycle", "model": "H2r", "color": "Black", "cc": 400, "price": 53000, "year": 2022},
    {"type": "Motorcycle", "model": "Unionstar", "color": "Red", "cc": 350, "price": 12000, "year": 2018},
    {"type": "Motorcycle", "model": "Superpower", "color": "Black", "cc": 70, "price": 94000, "year": 2021},
    {"type": "Motorcycle", "model": "Honda", "color": "Black", "cc": 110, "price": 9700, "year": 2022},
    {"type": "Motorcycle", "model": "Suzuki", "color": "Blue", "cc": 250, "price": 320000, "year": 2023},
    {"type": "Motorcycle", "model": "Crown", "color": "Black", "cc": 300, "price": 18000, "year": 2021},
    {"type": "Motorcycle", "model": "Ninja", "color": "Grey", "cc": 500, "price": 30000, "year": 2003},
    {"type": "Motorcycle", "model": "Superpower", "color": "Brown", "cc": 450, "price": 58000, "year": 2009},
    {"type": "Motorcycle", "model": "Superpower", "color": "White", "cc": 70, "price": 78200, "year": 2001},
    
    
    
]

car_data = [
    {"type": "Car", "model": "Corolla", "color": "Silver", "cc": 1800, "price": 130000, "year": 2021},
    {"type": "Car", "model": "Yaris", "color": "Black", "cc": 1200, "price": 435000, "year": 2022},
    {"type": "Car", "model": "Altis", "color": "Red", "cc": 1500, "price": 840000, "year": 2023},
    {"type": "Car", "model": "Audi", "color": "Grey", "cc": 2000, "price": 45000, "year": 2004},
    {"type": "Car", "model": "Vezel", "color": "Black", "cc": 2500, "price": 393000, "year": 2019},
    {"type": "Car", "model": "Cultus", "color": "Grey", "cc": 1800, "price": 55000, "year": 2021},
    {"type": "Car", "model": "MGHR", "color": "Silver", "cc": 2200, "price": 860000, "year": 2020},
    {"type": "Car", "model": "WagonR", "color": "White", "cc": 2000, "price": 65000, "year": 2018},
    {"type": "Car", "model": "Hiroof", "color": "Black", "cc": 2500, "price": 70000, "year": 2019},
    {"type": "Car", "model": "Civic", "color": "Gold", "cc": 2800, "price": 75000, "year": 2003},
    {"type": "Car", "model": "GTR", "color": "Black", "cc": 1200, "price": 35000, "year": 2022},
    {"type": "Car", "model": "Altis", "color": "Red", "cc": 1500, "price": 40000, "year": 2023},
    {"type": "Car", "model": "Audi", "color": "Grey", "cc": 2000, "price": 945000, "year": 2004},
    {"type": "Car", "model": "Etron", "color": "Black", "cc": 2500, "price": 5200, "year": 2015},
    {"type": "Car", "model": "Cultus", "color": "Grey", "cc": 1800, "price": 55000, "year": 2021},
    {"type": "Car", "model": "Alto", "color": "Silver", "cc": 2200, "price": 660000, "year": 2020},
    {"type": "Car", "model": "Yaris", "color": "Black", "cc": 1200, "price": 35000, "year": 2022},
    {"type": "Car", "model": "Tesla", "color": "Red", "cc": 1500, "price": 42000, "year": 2023},
    {"type": "Car", "model": "Audi", "color": "Grey", "cc": 2000, "price": 9500000, "year": 2004},
    {"type": "Car", "model": "MGHS", "color": "Grey", "cc": 1300, "price": 4500000, "year": 2004},
    {"type": "Car", "model": "Fx", "color": "Black", "cc": 2500, "price": 500000, "year": 2005},

]

# Convert data to a format suitable for the recommendation algorithm
corpus = [' '.join([item['type'], item['model'], item['color'], str(item['cc']), str(item['price']), str(item['year'])]) for item in motorcycle_data + car_data]

# Create TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

def get_recommendations(num_wheels, price, year, show_all=False):
    if num_wheels == 'Bike':
        data = motorcycle_data
    elif num_wheels == 'Car':
        data = car_data
    else:
        return []

    # Convert data to a format suitable for the recommendation algorithm
    corpus = [' '.join([item['type'], item['model'], item['color'], str(item['cc']), str(item['price']), str(item['year'])]) for item in data]

    if not corpus:
        return []

    user_input = ' '.join([str(num_wheels), ' '.join(corpus), str(price), str(year)])
    user_input_tfidf = tfidf_vectorizer.fit_transform([user_input] + corpus)
    cosine_similarities = linear_kernel(user_input_tfidf[0:1], user_input_tfidf).flatten()

    if len(cosine_similarities) > 0:
        related_indices = cosine_similarities.argsort()
        recommendations = [data[i] for i in related_indices if i < len(data)]
    else:
        recommendations = []

    # Refine sorting based on year and price
    recommendations.sort(key=lambda x: (abs(x['year'] - year), abs(x['price'] - price)))

    return recommendations

# Rest of the code remains unchanged
# Rest of the code remains unchanged
# GUI setup
def show_recommendations():
    num_wheels = num_wheels_combobox.get()
    price = price_entry.get()
    year = year_entry.get()

    if not num_wheels or not price or not year:
        messagebox.showerror("Error", "Please fill in all fields.")
        return

    if not num_wheels.isalpha() or not price.isdigit() or not year.isdigit():
        messagebox.showerror("Error", "Please select Vechile and enter valid numbers for Price, and Year.")
        return

    num_wheels = str(num_wheels)
    price = int(price)
    year = int(year)

    recommendations = get_recommendations(num_wheels, price, year)

    #
    text_column1.delete(1.0, tk.END)
    text_column2.delete(1.0, tk.END)
    text_column3.delete(1.0, tk.END)
    text_column4.delete(1.0, tk.END)
    text_column5.delete(1.0, tk.END)

    if recommendations:
        for i, recommendation in enumerate(recommendations[:5], start=1):
            text_column1.insert(tk.END, f"{i}. {recommendation['model']}\n")
            text_column2.insert(tk.END, f"{i}. {recommendation['color']}\n")
            text_column3.insert(tk.END, f"{i}. {recommendation['cc']}\n")
            text_column4.insert(tk.END, f"{i}. Rs. {recommendation['price']}\n")
            text_column5.insert(tk.END, f"{i}. {recommendation['year']}\n")
        if len(recommendations) > 5:
            show_more_button.config(state=tk.NORMAL)
        else:
            show_more_button.config(state=tk.DISABLED)
    else:
        messagebox.showinfo("Info", "No recommendations available.")

def show_more_recommendations():
    num_wheels = num_wheels_combobox.get()
    price = price_entry.get()
    year = year_entry.get()

    if not num_wheels or not price or not year:
        messagebox.showerror("Error", "Please fill in all fields.")
        return

    if not num_wheels.isalpha() or not price.isdigit() or not year.isdigit():
        messagebox.showerror("Error", "Please enter valid numbers for Number of Wheels, Price, and Year.")
        return

    num_wheels = str(num_wheels)
    price = int(price)
    year = int(year)

    recommendations = get_recommendations(num_wheels, price, year, show_all=True)

    # Clear previous results
    
    text_column1.delete(1.0, tk.END)
    text_column2.delete(1.0, tk.END)
    text_column3.delete(1.0, tk.END)
    text_column4.delete(1.0, tk.END)
    text_column5.delete(1.0, tk.END)

    if recommendations:
        for i, recommendation in enumerate(recommendations, start=1):
            text_column1.insert(tk.END, f"{i}. {recommendation['model']}\n")
            text_column2.insert(tk.END, f"{i}. {recommendation['color']}\n")
            text_column3.insert(tk.END, f"{i}. {recommendation['cc']}\n")
            text_column4.insert(tk.END, f"{i}. Rs.{recommendation['price']}\n")
            text_column5.insert(tk.END, f"{i}. {recommendation['year']}\n")
        show_more_button.config(state=tk.DISABLED)

# Create main window

# Create main window
root = tk.Tk()
root.title("Vehicle Recommendation System")
root.geometry("1440x800")  # Set the window size based on a 1920x1080 display resolution



# Center the window on the screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = (screen_width - 1440) // 2
y_coordinate = (screen_height - 800) // 2
root.geometry(f"1440x800+{x_coordinate}+{y_coordinate}")

# Set the background color
root.configure(bg='#259EA0')

#Seting heading
label=tk.Label(root, text="VEHICLE RECOMENDATION SYSTEM",font=("Helvetica", 35 ,"bold"), bg='#259EA0')
label.grid(row=0, column=1 ,padx=10, pady=10)
label.grid_rowconfigure(1, weight=1)
label.grid_columnconfigure(1, weight=1)


# Number of Wheels ComboBox
num_wheels_label = tk.Label(root, text="Select Vehic:", font=("Helvetica", 20,"bold"), bg='#259EA0')
num_wheels_label.grid(row=1, column=0, padx=10, pady=10)
num_wheels_values = ['Bike', 'Car']
num_wheels_combobox = ttk.Combobox(root, values=num_wheels_values, font=("Helvetica", 20))
num_wheels_combobox.grid(row=2, column=0, padx=10, pady=10)

# Price Entry
price_label = tk.Label(root, text="Enter Price:", font=("Helvetica", 20,"bold"), bg='#259EA0')
price_label.grid(row=3, column=0, padx=10, pady=10)
price_entry = tk.Entry(root, font=("Helvetica", 20))
price_entry.grid(row=4, column=0, padx=10, pady=10)

# Manufacture Year Entry
year_label = tk.Label(root, text="Enter Manufacture Year:", font=("Helvetica", 20,"bold"), bg='#259EA0')
year_label.grid(row=5, column=0, padx=10, pady=10)
year_entry = tk.Entry(root, font=("Helvetica", 20))
year_entry.grid(row=6, column=0, padx=10, pady=10)

# Recommendation Button
recommend_button = tk.Button(root, text="Get Recommendations", command=show_recommendations, font=("Helvetica", 20), fg='black',bg='#c8d0d9')
recommend_button.grid(row=7, column=0, pady=10)

# Show More Button
show_more_button = tk.Button(root, text="Show More", command=show_more_recommendations, state=tk.DISABLED, font=("Helvetica", 20), fg='black',bg='#c8d0d9')
show_more_button.grid(row=8, column=1, pady=10)

# Result Treeview with Scrollbar
result_frame = ttk.Frame(root)
result_frame.grid(row=1, column=1, columnspan=2,rowspan=7, pady=10)

label_column1 = tk.Label(result_frame, text="Model",font=("Helvetica", 15),cursor="circle")
label_column1.grid(row=0, column=0, padx=5, pady=5)

label_column2 = tk.Label(result_frame, text="Color",font=("Helvetica", 15),cursor="circle")
label_column2.grid(row=0, column=1, padx=5, pady=5)

label_column3 = tk.Label(result_frame, text="CC",font=("Helvetica", 15),cursor="circle")
label_column3.grid(row=0, column=2, padx=5, pady=5)

label_column4 = tk.Label(result_frame, text="Price",font=("Helvetica", 15),cursor="circle")
label_column4.grid(row=0, column=3, padx=5, pady=5)

label_column5 = tk.Label(result_frame, text="Years",font=("Helvetica", 15),cursor="circle")
label_column5.grid(row=0, column=4, padx=5, pady=5)


text_column1 = tk.Text(result_frame, height=16, width=15, font=("Helvetica", 20),cursor="circle")
text_column1.grid(row=1, column=0, padx=5, pady=5)

text_column2 = tk.Text(result_frame, height=16, width=12, font=("Helvetica", 20),cursor="circle")
text_column2.grid(row=1, column=1, padx=5, pady=5)

text_column3 = tk.Text(result_frame, height=16, width=12, font=("Helvetica", 20),cursor="circle")
text_column3.grid(row=1, column=2, padx=5, pady=5)

text_column4 = tk.Text(result_frame, height=16, width=15, font=("Helvetica", 20),cursor="circle")
text_column4.grid(row=1, column=3, padx=5, pady=5)

text_column5 = tk.Text(result_frame, height=16, width=12, font=("Helvetica", 20), state='normal',cursor="circle")
text_column5.grid(row=1, column=4, padx=5, pady=5)




# Start the GUI
root.mainloop()


# In[ ]:





# In[ ]:




