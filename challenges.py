#!/usr/bin/env python
# coding: utf-8

# # CHALLENGE 01
# # link : https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-projects/mean-variance-standard-deviation-calculator

# In[1]:


import numpy as np


# In[64]:


def calculate(list):
    if len(list) != 9 :
        raise ValueError('List must contain nine numbers.')
        
    a = np.array(list).reshape([3,3])
    r = {
        'mean' : [a.mean(0).tolist(),a.mean(1).tolist(),a.mean()],
        'variance' : [a.var(0).tolist(),a.var(1).tolist(),a.var()],
        'standard deviation' : [a.std(0).tolist(),a.std(1).tolist(),a.std()],
        'max' : [a.max(0).tolist(),a.max(1).tolist(),a.max()],
        'min' : [a.min(0).tolist(),a.min(1).tolist(),a.min()],
        'sum' : [a.sum(0).tolist(),a.sum(1).tolist(),a.sum()],
        }
    return r 


# In[65]:


print(calculate([0,1,2,3,4,5,6,7,8]))


# In[37]:


def calculate(arr):
    if len(arr) != 9:
        raise ValueError("List must contain nine numbers.")

    m = np.array(arr).reshape([3, 3])
    r = {
        "mean": [m.mean(0).tolist(), m.mean(1).tolist(), m.mean()],
        "variance": [m.var(0).tolist(), m.var(1).tolist(), m.var()],
        "standard deviation": [m.std(0).tolist(), m.std(1).tolist(), m.std()],
        "max": [m.max(0).tolist(), m.max(1).tolist(), m.max()],
        "min": [m.min(0).tolist(), m.min(1).tolist(), m.min()],
        "sum": [m.sum(0).tolist(), m.sum(1).tolist(), m.sum()],
    }
    return r


# # CHALLENGE 02
# # https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-projects/demographic-data-analyzer

# In[66]:


import pandas as pd
import numpy as np


# In[91]:


df = pd.read_csv(r'C:\Users\HP\Downloads\adult.data.csv')


# In[92]:


df.head()


# In[219]:


race_count = df['race'].value_counts()


# In[220]:


race_count


# In[122]:


avg_age = round(df[df.sex == 'Male'].age.mean(),1)


# In[123]:


avg_age


# In[227]:


bachelors_percentage = round(len(df[df['education']=='Bachelors'])/len(df) *100,1)


# In[228]:


bachelors_percentage


# In[233]:


q1 = df[df['education'].isin(['Bachelors', 'Masters', 'Doctorate'])]

higher_education_rich = round(len(q1[q1['salary'] == '>50K'])/len(q1) * 100, 1)


# In[234]:


higher_education_rich


# In[238]:


q2 = df[~df['education'].isin(['Bachelors', 'Masters', 'Doctorate'])]

lower_education_rich = round(len(q2[q2['salary'] == '>50K'])/len(q2) * 100, 1)


# In[239]:


lower_education_rich


# In[256]:


min_hrs = round(df['hours-per-week'].min(),1)


# In[257]:


min_hrs


# In[260]:


num_min_workers = len(df[df['hours-per-week'] == min_hrs])

rich_percentage = round(len(df[(df['hours-per-week'] == min_hrs) & (df['salary'] == '>50K')]) / num_min_workers * 100, 1)


# In[262]:


rich_percentage


# In[267]:


highest_earning_country = (df.loc[df['salary'] == ">50K", 'native-country'].value_counts() / df['native-country'].value_counts()).fillna(0).sort_values(ascending=False).index[0]
highest_earning_country_percentage = round(len(df[(df['native-country'] == highest_earning_country) & (df['salary'] == '>50K')]) / len(df[df['native-country'] == highest_earning_country]) * 100, 1)


# In[268]:


highest_earning_country


# In[269]:


highest_earning_country_percentage


# In[274]:


popular_occupation = df[(df['salary'] == ">50K") & (df['native-country'] == "India")]["occupation"].value_counts().index[0]


# In[275]:


popular_occupation


# In[278]:


print('People of Each race :\n' ,race_count),
print('Avg age of men :\n', avg_age),
print('Percentage of people who have a Bachelors degree:\n', bachelors_percentage),
print('Percentage of people with advanced education making more than 50K:\n', higher_education_rich),
print('Percentage of people without advanced education making more than 50K:\n', lower_education_rich),
print('Minimum number of hours a person works per week:\n', min_hrs),
print('Percentage of the people who work the minimum number of hours per week have a salary of more than 50K:\n', rich_percentage),
print('Country has the highest percentage of people that earn >50K:\n', highest_earning_country)
print('Country has the highest percentage of people that earn >50K percentage :\n', highest_earning_country_percentage),
print('Most popular occupation for those who earn >50K in India :\n', popular_occupation)


# # CHALLENGE 03
# # https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-projects/medical-data-visualizer

# In[1]:


import pandas as pd
import numpy as np


# In[129]:


df = pd.read_csv(r'C:\Users\HP\Downloads\medical_examination.csv')


# In[130]:


df.head()


# In[131]:


bmi = df['weight'] / ((df['height'] / 100) ** 2)
overweight = []
for i in bmi:
    if i > 25:
        overweight.append(1)
    if i <= 25:
        overweight.append(0)
df['overweight'] = overweight

df.head()


# In[64]:


df['cholesterol'] = df['cholesterol'].apply(lambda x : 0 if x == 1 else 1)


# In[66]:


df['gluc'] = df['gluc'].apply(lambda x : 0 if x == 1 else 1)


# In[70]:


df.loc[df['cholesterol'] == 1, 'cholesterol'] = 0
df.loc[df['cholesterol'] >= 2, 'cholesterol'] = 1
df.loc[df['gluc'] == 1, 'gluc'] = 0
df.loc[df['gluc'] >= 2, 'gluc'] = 1


# In[71]:


df.head()


# In[46]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[47]:


df.dtypes


# In[102]:


df.astype(float).head()


# In[109]:


df_cat = df.melt(id_vars = 'cardio', 
                     value_vars = ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], 
                     value_name='value')
fig = sns.catplot(data=df_cat, kind='count', x='variable', hue='value', col='cardio')
fig.set_axis_labels('variable','totals')
fig.savefig('catplot.png')


# In[116]:


indexAge = df[ (df['ap_lo'] <= df['ap_hi']) | 
              (df['height'] <= df['height'].quantile(0.025)) |
              (df['height'] > df['height'].quantile(0.975)) |
              (df['weight'] <= df['weight'].quantile(0.025)) |
              (df['weight'] > df['weight'].quantile(0.975))].index
df.drop(indexAge , inplace=True)


# In[117]:


df.head()


# In[132]:


def draw_heat_map():
    # Clean the data
    df_heat = \
        df[(df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(df_heat.corr(), dtype=bool))



    # Set up the matplotlib figure
    fig, ax = plt.subplots()

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(data=corr, 
                annot=True, 
                fmt=".1f", 
                linewidth=.5, 
                mask=mask, 
                annot_kws={'fontsize':6}, 
                cbar_kws={"shrink": .7}, 
                square=False, 
                center=0, 
                vmax=0.30);


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig

draw_heat_map()


# # CHALLENGE 04
# # https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-projects/page-view-time-series-visualizer

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# In[2]:


df = pd.read_csv(r'C:\Users\HP\Downloads\fcc-forum-pageviews.csv',
                parse_dates = [0],
                index_col = ['date']
                )


# In[3]:


df_clean = df[(df['value'] > df['value'].quantile(0.025)) & (df['value'] < df['value'].quantile(0.975))]    
df_clean.head()


# In[5]:


df_bar = df.copy()
df_bar['year'] = df_bar.index.year
df_bar['month'] = df_bar.index.month_name()
df_bar = df_bar.groupby(['year','month']).mean().unstack()
bar = df_bar.plot(kind = 'bar', legend = True, figsize = (14,6)).figure
plt.title('Average Daily Page Views, Grouped by Year and Month')
plt.xlabel('Years')
plt.ylabel('Average Page Views')
plt.legend(title = 'Month', labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
plt.show()


# In[11]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Import data (Make sure to parse dates. Consider setting index column to 'date'.)
df = pd.read_csv(r'C:\Users\HP\Downloads\fcc-forum-pageviews.csv', index_col='date', parse_dates=['date'])

# Clean data
df = df[(df['value'] > df['value'].quantile(0.025)) & (df['value'] < df['value'].quantile(0.975))] 


def draw_line_plot():
    # Draw line plot
    fig = df.plot.line(figsize = (12,6))
    plt.title('Daily freeCodeCamp Forum Page Views 5/2016-12/2019')
    plt.xlabel('Date')
    plt.xticks(rotation = 0)
    plt.ylabel('Page Views')
    fig = fig.figure



    # Save image and return fig (don't change this part)
    fig.savefig('line_plot.png')
    return fig

def draw_bar_plot():
    # Copy and modify data for monthly bar plot
    df_bar = df.copy()
    df_bar['year'] = df_bar.index.year
    df_bar['month'] = df_bar.index.month_name()
    df_bar = df_bar.groupby(['year','month']).mean().unstack()

    # Draw bar plot
    fig = df_bar.plot(kind = 'bar', legend = True, figsize = (14,6)).figure
    plt.title('Average Daily Page Views, Grouped by Year and Month')
    plt.xlabel('Years')
    plt.ylabel('Average Page Views')
    plt.legend(title = 'Month', labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
  






    # Save image and return fig (don't change this part)
    fig.savefig('bar_plot.png')
    fig.show()

def draw_box_plot():
    # Prepare data for box plots (this part is done!)
    df_box = df.copy()
    df_box.reset_index(inplace=True)
    df_box['year'] = [d.year for d in df_box.date]
    df_box['month'] = [d.strftime('%b') for d in df_box.date]
    mon_order = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    # Draw box plots (using Seaborn)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.boxplot(ax=ax1, data=df_box, x=df_box["year"], y=df_box["value"])
    ax1.set(
        xlabel="Year", ylabel="Page Views", title="Year-wise Box Plot (Trend)"
    )

    sns.boxplot(
        ax=ax2,
        data=df_box,
        x=df_box["month"],
        y=df_box["value"],
        order=[
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )

    ax2.set_title("Month-wise Box Plot (Seasonality)")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Page Views")
    y_ticks = [
        "0",
        "20000",
        "40000",
        "60000",
        "80000",
        "100000",
        "120000",
        "140000",
        "160000",
        "180000",
        "200000",
    ]
    ax1.yaxis.set_major_locator(mticker.FixedLocator([int(s) for s in y_ticks]))
    ax1.set_yticklabels(y_ticks)
    
    




    # Save image and return fig (don't change this part)
    fig.savefig('box_plot.png')
    return fig


# # CHALLENGE 05
# # https://www.freecodecamp.org/learn/data-analysis-with-python/data-analysis-with-python-projects/sea-level-predictor

# In[2]:


df = pd.read_csv(r'C:\Users\HP\Downloads\epa-sea-level.csv')
df.head()


# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
x = df['Year']
y = df['CSIRO Adjusted Sea Level']
plt.figure(figsize = (14,6))
plt.scatter(x,y, alpha=0.5)
plt.colorbar()
plt.show()


# In[24]:


plt.scatter(df['Year'], df['CSIRO Adjusted Sea Level'])


# In[25]:


lr_1880_2012 = linregress(df['Year'], df['CSIRO Adjusted Sea Level'])
plt.plot(range(1880, 2051, 1), lr_1880_2012.slope*range(1880, 2051, 1) + lr_1880_2012.intercept)
plt.show()


# In[30]:


plt.scatter(df['Year'], df['CSIRO Adjusted Sea Level'])
lr_1880_2012 = linregress(df['Year'], df['CSIRO Adjusted Sea Level'])
plt.plot(range(1880, 2051, 1), lr_1880_2012.slope*range(1880, 2051, 1) + lr_1880_2012.intercept)
lr_2000_2012 = linregress(df.query('Year >= 2000')['Year'],
                          df.query('Year >= 2000')['CSIRO Adjusted Sea Level'])
plt.plot(range(2000, 2051, 1), lr_2000_2012.slope*range(2000, 2051, 1) + lr_2000_2012.intercept)
plt.title('Rise in Sea Level')
plt.ylabel('Sea Level (inches)')
plt.xlabel('Year')
plt.show()


# # VARIOUS Algorithms 

# In[1]:


#Cipher Building
text = 'mrttaqrhknsw ih puggrur'
custom_key = 'happycoding'

def vigenere(message, key, direction=1):
    key_index = 0
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    final_message = ''

    for char in message.lower():

        # Append any non-letter character to the message
        if not char.isalpha():
            final_message += char
        else:        
            # Find the right key character to encode/decode
            key_char = key[key_index % len(key)]
            key_index += 1

            # Define the offset and the encrypted/decrypted letter
            offset = alphabet.index(key_char)
            index = alphabet.find(char)
            new_index = (index + offset*direction) % len(alphabet)
            final_message += alphabet[new_index]
    
    return final_message

def encrypt(message, key):
    return vigenere(message, key)
    
def decrypt(message, key):
    return vigenere(message, key, -1)

print(f'\nEncrypted text: {text}')
print(f'Key: {custom_key}')
decryption = decrypt(text, custom_key)
print(f'\nDecrypted text: {decryption}\n')


# In[15]:


#Luhn Algorithm
def verify_card_number(card_number):
   sum_of_odd_digits = 0
   card_number_reversed = card_number[::-1]
   print('reversed card number:',card_number_reversed)
   odd_digits = card_number_reversed[::2]
   print('odd placed digits:',odd_digits)

   for digit in odd_digits:
       sum_of_odd_digits += int(digit)
       print('sum of odd placed digits:',sum_of_odd_digits)

   sum_of_even_digits = 0
   even_digits = card_number_reversed[1::2]
   print('even placed digits:',even_digits)
   for digit in even_digits:
       number = int(digit) * 2
       print('doubled even placed digits:',number)
       if number >= 10:
           number = (number // 10) + (number % 10)
           print('sum of digits if double of even placed digits are more than 10:',number)
       sum_of_even_digits += number
       print('sum of even placed digits:',sum_of_even_digits)
   total = sum_of_odd_digits + sum_of_even_digits
   print('total sum of odd and even placed digits:',total)
   return total % 10 == 0

def main():
   card_number = '4111-1111-4555-1142'
   
   card_translation = str.maketrans({'-': '', ' ': ''})
   print(card_translation)
   translated_card_number = card_number.translate(card_translation)
   print(translated_card_number)
   if verify_card_number(translated_card_number):
       print('VALID!')
   else:
       print('INVALID!')

main()


# In[1]:


# Build a Budget App Project
def add_expense(expenses, amount, category):
    expenses.append({'amount': amount, 'category': category})
    
def print_expenses(expenses):
    for expense in expenses:
        print(f'Amount: {expense["amount"]}, Category: {expense["category"]}')
    
def total_expenses(expenses):
    return sum(map(lambda expense: expense['amount'], expenses))
    
def filter_expenses_by_category(expenses, category):
    return filter(lambda expense: expense['category'] == category, expenses)
    

def main():
    expenses = []
    while True:
        print('\nExpense Tracker')
        print('1. Add an expense')
        print('2. List all expenses')
        print('3. Show total expenses')
        print('4. Filter expenses by category')
        print('5. Exit')
       
        choice = input('Enter your choice: ')

        if choice == '1':
            amount = float(input('Enter amount: '))
            category = input('Enter category: ')
            add_expense(expenses, amount, category)

        elif choice == '2':
            print('\nAll Expenses:')
            print_expenses(expenses)
    
        elif choice == '3':
            print('\nTotal Expenses: ', total_expenses(expenses))
    
        elif choice == '4':
            category = input('Enter category to filter: ')
            print(f'\nExpenses for {category}:')
            expenses_from_category = filter_expenses_by_category(expenses, category)
            print_expenses(expenses_from_category)
    
        elif choice == '5':
            print('Exiting the program.')
            break

main()


# In[4]:


#Case converter
def convert_to_snake_case(pascal_or_camel_cased_string):
    snake_cased_char_list = []
    for char in pascal_or_camel_cased_string:
        if char.isupper():
            converted_character = '_' + char.lower()
            snake_cased_char_list.append(converted_character)
        else:
            snake_cased_char_list.append(char)
    snake_cased_string = ''.join(snake_cased_char_list)
    clean_snake_cased_string = snake_cased_string.strip('_')

    return clean_snake_cased_string

def main():
    print(convert_to_snake_case('aLongAndComplexString'))
main()


# In[5]:


def convert_to_snake_case(pascal_or_camel_cased_string):

    snake_cased_char_list = [
        '_' + char.lower() if char.isupper()
        else char
        for char in pascal_or_camel_cased_string
    ]

    return ''.join(snake_cased_char_list).strip('_')

def main():
    print(convert_to_snake_case('IAmAPascalCasedString'))
main()


# In[8]:


# Root using bisection 
def square_root_bisection(square_target, tolerance=1e-7, max_iterations=100):
    if square_target < 0:
        raise ValueError('Square root of negative number is not defined in real numbers')
    if square_target == 1:
        root = 1
        print(f'The square root of {square_target} is 1')
    elif square_target == 0:
        root = 0
        print(f'The square root of {square_target} is 0')

    else:
        low = 0
        high = max(1, square_target)
        root = None
        
        for _ in range(max_iterations):
            mid = (low + high) / 2
            square_mid = mid**2

            if abs(square_mid - square_target) < tolerance:
                root = mid
                break

            elif square_mid < square_target:
                low = mid
            else:
                high = mid

        if root is None:
            print(f"Failed to converge within {max_iterations} iterations.")
    
        else:   
            print(f'The square root of {square_target} is approximately {root}')
    
    return root

N = 16987768975635272
square_root_bisection(N)


# In[3]:


# arithmetic arranger
def arithmetic_arranger(problems, show_answers=False):
  if len(problems) > 5:
      return "Error: Too many problems."
  
  first_line = ""
  second_line = ""
  dash_line = ""
  answer_line = ""
  
  for problem in problems:
      num1, operator, num2 = problem.split()
  
      if operator not in ['+', '-']:
          return "Error: Operator must be '+' or '-'."
  
      if not (num1.isdigit() and num2.isdigit()):
          return "Error: Numbers must only contain digits."
  
      if len(num1) > 4 or len(num2) > 4:
          return "Error: Numbers cannot be more than four digits."
  
      length = max(len(num1), len(num2)) + 2
      first_line += num1.rjust(length) + "    "
      second_line += operator + num2.rjust(length - 1) + "    "
      dash_line += "-" * length + "    "
  
      if show_answers:
          if operator == '+':
              answer = str(int(num1) + int(num2))
          else:
              answer = str(int(num1) - int(num2))
          answer_line += answer.rjust(length) + "    "
      arranged_problems = first_line.rstrip() + '\n' + second_line.rstrip() + "\n" + dash_line.rstrip() 
      
      if show_answers:
          arranged_problems = first_line.rstrip() + '\n' + second_line.rstrip() + "\n" + dash_line.rstrip() + "\n" + answer_line.rstrip()
  
  return arranged_problems


# In[4]:


arithmetic_arranger(["32 - 698", "1 - 3801", "45 + 43", "123 + 49", "988 + 40"], True)


# In[3]:


#Generate Password
import re
import secrets
import string


def generate_password(length=16, nums=1, special_chars=1, uppercase=1, lowercase=1):

    # Define the possible characters for the password
    letters = string.ascii_letters
    digits = string.digits
    symbols = string.punctuation

    # Combine all characters
    all_characters = letters + digits + symbols

    while True:
        password = ''
        # Generate password
        for _ in range(length):
            password += secrets.choice(all_characters)
        
        constraints = [
            (nums, r'\d'),
            (special_chars, fr'[{symbols}]'),
            (uppercase, r'[A-Z]'),
            (lowercase, r'[a-z]')
        ]

        # Check constraints        
        if all(
            constraint <= len(re.findall(pattern, password))
            for constraint, pattern in constraints
        ):
            break
    
    return password
    
new_password = generate_password()
print('Generated password:', new_password)


# In[5]:


#shortest path
my_graph = {
    'A': [('B', 5), ('C', 3), ('E', 11)],
    'B': [('A', 5), ('C', 1), ('F', 2)],
    'C': [('A', 3), ('B', 1), ('D', 1), ('E', 5)],
    'D': [('C',1 ), ('E', 9), ('F', 3)],
    'E': [('A', 11), ('C', 5), ('D', 9)],
    'F': [('B', 2), ('D', 3)]
}

def shortest_path(graph, start, target = ''):
    unvisited = list(graph)
    distances = {node: 0 if node == start else float('inf') for node in graph}
    paths = {node: [] for node in graph}
    paths[start].append(start)
    
    while unvisited:
        current = min(unvisited, key=distances.get)
        for node, distance in graph[current]:
            if distance + distances[current] < distances[node]:
                distances[node] = distance + distances[current]
                if paths[node] and paths[node][-1] == node:
                    paths[node] = paths[current][:]
                else:
                    paths[node].extend(paths[current])
                paths[node].append(node)
        unvisited.remove(current)
    
    targets_to_print = [target] if target else graph
    for node in targets_to_print:
        if node == start:
            continue
        print(f'\n{start}-{node} distance: {distances[node]}\nPath: {" -> ".join(paths[node])}')
    
    return distances, paths
    
shortest_path(my_graph, 'A','F')


# In[1]:


#Hanoi Puzzle
NUMBER_OF_DISKS = 4
number_of_moves = 2 ** NUMBER_OF_DISKS - 1
rods = {
    'A': list(range(NUMBER_OF_DISKS, 0, -1)),
    'B': [],
    'C': []
}

def make_allowed_move(rod1, rod2):    
    forward = False
    if not rods[rod2]:
        forward = True
    elif rods[rod1] and rods[rod1][-1] < rods[rod2][-1]:
        forward = True              
    if forward:
        print(f'Moving disk {rods[rod1][-1]} from {rod1} to {rod2}')
        rods[rod2].append(rods[rod1].pop())
    else:
        print(f'Moving disk {rods[rod2][-1]} from {rod2} to {rod1}')
        rods[rod1].append(rods[rod2].pop())
    
    # display our progress
    print(rods, '\n')

def move(n, source, auxiliary, target):
    # display starting configuration
    print(rods, '\n')
    for i in range(number_of_moves):
        remainder = (i + 1) % 3
        if remainder == 1:
            if n % 2 != 0:
                print(f'Move {i + 1} allowed between {source} and {target}')
                make_allowed_move(source, target)
            else:
                print(f'Move {i + 1} allowed between {source} and {auxiliary}')
                make_allowed_move(source, auxiliary)
        elif remainder == 2:
            if n % 2 != 0:
                print(f'Move {i + 1} allowed between {source} and {auxiliary}')
                make_allowed_move(source, auxiliary)
            else:
                print(f'Move {i + 1} allowed between {source} and {target}')
                make_allowed_move(source, target)
        elif remainder == 0:
            print(f'Move {i + 1} allowed between {auxiliary} and {target}')
            make_allowed_move(auxiliary, target)           

# initiate call from source A to target C with auxiliary B
move(NUMBER_OF_DISKS, 'A', 'B', 'C')


# In[2]:


NUMBER_OF_DISKS = 3
A = list(range(NUMBER_OF_DISKS, 0, -1))
B = []
C = []

def move(n, source, auxiliary, target):
    if n <= 0:
        return
        # move n - 1 disks from source to auxiliary, so they are out of the way
    move(n - 1, source, target, auxiliary)
        
        # move the nth disk from source to target
    target.append(source.pop())
        
        # display our progress
    print(A, B, C, '\n')
        
        # move the n - 1 disks that we left on auxiliary onto target
    move(n - 1,  auxiliary, source, target)
              
# initiate call from source A to target C with auxiliary B
move(NUMBER_OF_DISKS, A, B, C)


# In[4]:


#Merge Sort
def merge_sort(array):
    if len(array) <= 1:
        return
    
    middle_point = len(array) // 2
    left_part = array[:middle_point]
    right_part = array[middle_point:]

    merge_sort(left_part)
    merge_sort(right_part)

    left_array_index = 0
    right_array_index = 0
    sorted_index = 0

    while left_array_index < len(left_part) and right_array_index < len(right_part):
        if left_part[left_array_index] < right_part[right_array_index]:
            array[sorted_index] = left_part[left_array_index]
            left_array_index += 1
        else:
            array[sorted_index] = right_part[right_array_index]
            right_array_index += 1
        sorted_index += 1

    while left_array_index < len(left_part):
        array[sorted_index] = left_part[left_array_index]
        left_array_index += 1
        sorted_index += 1
    
    while right_array_index < len(right_part):
        array[sorted_index] = right_part[right_array_index]
        right_array_index += 1
        sorted_index += 1


if __name__ == '__main__':
    numbers = [4, 10, 6, 14, 2, 1, 8, 5]
    print('Unsorted array: ')
    print(numbers)
    merge_sort(numbers)
    print('Sorted array: ' + str(numbers))


# In[1]:


def add_time(current_time, duration, day=None):

  # define helper variable
  days_note = ""
  days_name = ""
  week = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
  changing_format = {"AM":"PM", "PM":"AM"}

  # get time detail (hour, minute, and format) for each time params
  current_time_hour = int(current_time.split(" ")[0].split(":")[0])
  current_time_format = current_time.split(" ")[1]
  duration_hour = int(duration.split(":")[0])
  next_time_minutes = int(current_time.split(" ")[0].split(":")[1]) + int(duration.split(":")[1])

  # check if minute more than 60 and convert to hour if minute > 60
  if next_time_minutes > 60:
    next_time_minutes -= 60
    duration_hour += 1

  # calcutate next time hour, format and total days
  count_hours = (current_time_hour + duration_hour) % 12
  count_days = (current_time_hour + duration_hour) // 24
  count_format = (current_time_hour + duration_hour) % 24
  
  days_later = (count_days + 1) if (current_time_hour + duration_hour) >= 12 and current_time_format == "PM" else count_days
  next_time_hours = count_hours if count_hours != 0 else 12
  next_time_format = changing_format[current_time_format] if count_format >= 12 else current_time_format

  # set hour to 0 if current time format == PM and current time hour + duration hour == 12 
  next_time_hours = 0 if (current_time_hour + duration_hour) == 12 and current_time_format == "PM" else next_time_hours
  
  # set days later note
  if days_later > 0:
    days_note = " (next day)" if days_later == 1 else f" ({days_later} days later)"

  # get current day
  if day:
    current_day_index = week.index(day.capitalize())
    reset_week = week[current_day_index:] + week[:current_day_index]
    next_day_index = (days_later % 7) if days_later > len(reset_week) else days_later
    days_name = f", {reset_week[next_day_index]}"

  return f"{next_time_hours}:{str(next_time_minutes).zfill(2)} {next_time_format}{days_name}{days_note}"


# In[2]:


add_time('3:50 PM', '1:20', 'Monday')


# In[3]:


class Board:
    def __init__(self, board):
        self.board = board

    def __str__(self):
        board_str = ''
        for row in self.board:
            row_str = [str(i) if i else '*' for i in row]
            board_str += ' '.join(row_str)
            board_str += '\n'
        return board_str

    def find_empty_cell(self):
        for row, contents in enumerate(self.board):
            try:
                col = contents.index(0)
                return row, col
            except ValueError:
                pass
        return None

    def valid_in_row(self, row, num):
        return num not in self.board[row]

    def valid_in_col(self, col, num):
        return all(self.board[row][col] != num for row in range(9))

    def valid_in_square(self, row, col, num):
        row_start = (row // 3) * 3
        col_start = (col // 3) * 3
        for row_no in range(row_start, row_start + 3):
            for col_no in range(col_start, col_start + 3):
                if self.board[row_no][col_no] == num:
                    return False
        return True

    def is_valid(self, empty, num):
        row, col = empty
        valid_in_row = self.valid_in_row(row, num)
        valid_in_col = self.valid_in_col(col, num)
        valid_in_square = self.valid_in_square(row, col, num)
        return all([valid_in_row, valid_in_col, valid_in_square])

    def solver(self):
        if (next_empty := self.find_empty_cell()) is None:
            return True
        for guess in range(1, 10):
            if self.is_valid(next_empty, guess):
                row, col = next_empty
                self.board[row][col] = guess
                if self.solver():
                    return True
                self.board[row][col] = 0
        return False

def solve_sudoku(board):
    gameboard = Board(board)
    print(f'Puzzle to solve:\n{gameboard}')
    if gameboard.solver():
        print(f'Solved puzzle:\n{gameboard}')
    else:
        print('The provided puzzle is unsolvable.')
    return gameboard

puzzle = [
  [0, 0, 2, 0, 0, 8, 0, 0, 0],
  [0, 0, 0, 0, 0, 3, 7, 6, 2],
  [4, 3, 0, 0, 0, 0, 8, 0, 0],
  [0, 5, 0, 0, 3, 0, 0, 9, 0],
  [0, 4, 0, 0, 0, 0, 0, 2, 6],
  [0, 0, 0, 4, 6, 7, 0, 0, 0],
  [0, 8, 6, 7, 0, 4, 0, 0, 0],
  [0, 0, 0, 5, 1, 9, 0, 0, 8],
  [1, 7, 0, 0, 0, 6, 0, 0, 5]
]
solve_sudoku(puzzle)


# In[11]:


#def __str__(self):
picture = ''
for w in picture:
    w_str = [str(i) if i else '*' for i in picture]
    picture += '*'.join(w_str)
    picture += '\n'
    print(picture)


# In[1]:


class TreeNode:

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

    def __str__(self):
        return str(self.key)

class BinarySearchTree:

    def __init__(self):
        self.root = None

    def _insert(self, node, key):
        if node is None:
            return TreeNode(key)

        if key < node.key:
            node.left = self._insert(node.left, key)
        elif key > node.key:

            node.right = self._insert(node.right, key)
        return node

    def insert(self, key):
        self.root = self._insert(self.root, key)
        
    def _search(self, node, key):
        if node is None or node.key == key:
            return node
        if key < node.key:
            return self._search(node.left, key)
        return self._search(node.right, key)
    
    def search(self, key):
        return self._search(self.root, key)

    def _delete(self, node, key):
        if node is None:
            return node
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key) 
        else:
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left   
            
            node.key = self._min_value(node.right)
            node.right = self._delete(node.right, node.key)   
        
        return node

    def delete(self, key):
        self.root = self._delete(self.root, key)

    def _min_value(self, node):
        while node.left is not None:
            node = node.left
        return node.key

    def _inorder_traversal(self, node, result):
        if node:
            self._inorder_traversal(node.left, result)
            result.append(node.key)
            self._inorder_traversal(node.right, result)

    def inorder_traversal(self):
        result = []
        self._inorder_traversal(self.root, result)
        return result

bst = BinarySearchTree()
nodes = [50, 30, 20, 40, 70, 60, 80]

for node in nodes:
    bst.insert(node)

print('Search for 80:', bst.search(80))

print("Inorder traversal:", bst.inorder_traversal())

bst.delete(40)

print("Search for 40:", bst.search(40))
print('Inorder traversal after deleting 40:',bst.inorder_traversal())


# In[5]:


class Category:
    def __init__(self, description):
        self.description = description
        self.ledger = []
        self.__balance = 0.0

    def __repr__(self):
        header = self.description.center(30, "*") + "\n"
        ledger = ""
        for item in self.ledger:
            # format description and amount
            line_description = "{:<23}".format(item["description"])
            line_amount = "{:>7.2f}".format(item["amount"])
            # Truncate ledger description and amount to 23 and 7 characters respectively
            ledger += "{}{}\n".format(line_description[:23], line_amount[:7])
        total = "Total: {:.2f}".format(self.__balance)
        return header + ledger + total

    def deposit(self, amount, description=""):
        self.ledger.append({"amount": amount, "description": description})
        self.__balance += amount

    def withdraw(self, amount, description=""):
        if self.__balance - amount >= 0:
            self.ledger.append({"amount": -1 * amount, "description": description})
            self.__balance -= amount
            return True
        else:
            return False

    def get_balance(self):
        return self.__balance

    def transfer(self, amount, category_instance):
        if self.withdraw(amount, "Transfer to {}".format(category_instance.description)):
            category_instance.deposit(amount, "Transfer from {}".format(self.description))
            return True
        else:
            return False

    def check_funds(self, amount):
        if self.__balance >= amount:
            return True
        else:
            return False


def create_spend_chart(categories):
    spent_amounts = []
    # Get total spent in each category
    for category in categories:
        spent = 0
        for item in category.ledger:
            if item["amount"] < 0:
                spent += abs(item["amount"])
        spent_amounts.append(round(spent, 2))

    # Calculate percentage rounded down to the nearest 10
    total = round(sum(spent_amounts), 2)
    spent_percentage = list(map(lambda amount: int((((amount / total) * 10) // 1) * 10), spent_amounts))

    # Create the bar chart substrings
    header = "Percentage spent by category\n"

    chart = ""
    for value in reversed(range(0, 101, 10)):
        chart += str(value).rjust(3) + '|'
        for percent in spent_percentage:
            if percent >= value:
                chart += " o "
            else:
                chart += "   "
        chart += " \n"

    footer = "    " + "-" * ((3 * len(categories)) + 1) + "\n"
    descriptions = list(map(lambda category: category.description, categories))
    max_length = max(map(lambda description: len(description), descriptions))
    descriptions = list(map(lambda description: description.ljust(max_length), descriptions))
    for x in zip(*descriptions):
        footer += "    " + "".join(map(lambda s: s.center(3), x)) + " \n"

    return (header + chart + footer).rstrip("\n")




# In[1]:


class R2Vector:
    def __init__(self, *, x, y):
        self.x = x
        self.y = y

    def norm(self):
        return sum(val**2 for val in vars(self).values())**0.5

    def __str__(self):
        return str(tuple(getattr(self, i) for i in vars(self)))

    def __repr__(self):
        arg_list = [f'{key}={val}' for key, val in vars(self).items()]
        args = ', '.join(arg_list)
        return f'{self.__class__.__name__}({args})'

    def __add__(self, other):
        if type(self) != type(other):
            return NotImplemented
        kwargs = {i: getattr(self, i) + getattr(other, i) for i in vars(self)}
        return self.__class__(**kwargs)

    def __sub__(self, other):
        if type(self) != type(other):
            return NotImplemented
        kwargs = {i: getattr(self, i) - getattr(other, i) for i in vars(self)}
        return self.__class__(**kwargs)

    def __mul__(self, other):
        if type(other) in (int, float):
            kwargs = {i: getattr(self, i) * other for i in vars(self)}
            return self.__class__(**kwargs)        
        elif type(self) == type(other):
            args = [getattr(self, i) * getattr(other, i) for i in vars(self)]
            return sum(args)            
        return NotImplemented

    def __eq__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return all(getattr(self, i) == getattr(other, i) for i in vars(self))
        
    def __ne__(self, other):
        return not self == other

    def __lt__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.norm() < other.norm()

    def __gt__(self, other):
        if type(self) != type(other):
            return NotImplemented
        return self.norm() > other.norm()

    def __le__(self, other):
        return not self > other

    def __ge__(self, other):
        return not self < other

class R3Vector(R2Vector):
    def __init__(self, *, x, y, z):
        super().__init__(x=x, y=y)
        self.z = z
        
    def cross(self, other):
        if type(self) != type(other):
            return NotImplemented
        kwargs = {
            'x': self.y * other.z - self.z * other.y,
            'y': self.z * other.x - self.x * other.z,
            'z': self.x * other.y - self.y * other.x
        }
        
        return self.__class__(**kwargs)
v1 = R3Vector(x=2, y=3, z=1)
v2 = R3Vector(x=0.5, y=1.25, z=2)
print(f'v1 = {v1}')
print(f'v2 = {v2}')
v3 = v1 + v2
print(f'v1 + v2 = {v3}')
v4 = v1 - v2
print(f'v1 - v2 = {v4}')
v5 = v1 * v2
print(f'v1 * v2 = {v5}')
v6 = v1.cross(v2)
print(f'v1 x v2 = {v6}')


# In[1]:


from abc import ABC, abstractmethod
import re


class Equation(ABC):
    degree: int
    type: str
  
    def __init__(self, *args):
        if (self.degree + 1) != len(args):
            raise TypeError(
                f"'Equation' object takes {self.degree + 1} positional arguments but {len(args)} were given"
            )
        if any(not isinstance(arg, (int, float)) for arg in args):
            raise TypeError("Coefficients must be of type 'int' or 'float'")
        if args[0] == 0:
            raise ValueError("Highest degree coefficient must be different from zero")
        self.coefficients = {(len(args) - n - 1): arg for n, arg in enumerate(args)}

    def __init_subclass__(cls):
        if not hasattr(cls, "degree"):
            raise AttributeError(
                f"Cannot create '{cls.__name__}' class: missing required attribute 'degree'"
            )
        if not hasattr(cls, "type"):
            raise AttributeError(
                f"Cannot create '{cls.__name__}' class: missing required attribute 'type'"
            )

    def __str__(self):
        terms = []
        for n, coefficient in self.coefficients.items():
            if not coefficient:
                continue
            if n == 0:
                terms.append(f'{coefficient:+}')
            elif n == 1:
                terms.append(f'{coefficient:+}x')
            else:
                terms.append(f"{coefficient:+}x**{n}")
        equation_string = ' '.join(terms) + ' = 0'
        return re.sub(r"(?<!\d)1(?=x)", "", equation_string.strip("+"))        

    @abstractmethod
    def solve(self):
        pass
        
    @abstractmethod
    def analyze(self):
        pass


class LinearEquation(Equation):
    degree = 1
    type = 'Linear Equation'
    
    def solve(self):
        a, b = self.coefficients.values()
        x = -b / a
        return [x]

    def analyze(self):
        slope, intercept = self.coefficients.values()
        return {'slope': slope, 'intercept': intercept}


class QuadraticEquation(Equation):
    degree = 2
    type = 'Quadratic Equation'

    def __init__(self, *args):
        super().__init__(*args)
        a, b, c = self.coefficients.values()
        self.delta = b**2 - 4 * a * c

    def solve(self):
        if self.delta < 0:
            return []
        a, b, _ = self.coefficients.values()
        x1 = (-b + (self.delta) ** 0.5) / (2 * a)
        x2 = (-b - (self.delta) ** 0.5) / (2 * a)
        if self.delta == 0:
            return [x1]

        return [x1, x2]

    def analyze(self):
        a, b, c = self.coefficients.values()
        x = -b / (2 * a)
        y = a * x**2 + b * x + c
        if a > 0:
            concavity = 'upwards'
            min_max = 'min'
        else:
            concavity = 'downwards'
            min_max = 'max'
        return {'x': x, 'y': y, 'min_max': min_max, 'concavity': concavity}


def solver(equation):
    if not isinstance(equation, Equation):
        raise TypeError("Argument must be an Equation object")

    output_string = f'\n{equation.type:-^24}'
    output_string += f'\n\n{equation!s:^24}\n\n'
    output_string += f'{"Solutions":-^24}\n\n'
    results = equation.solve()
    match results:
        case []:
            result_list = ['No real roots']
        case [x]:
            result_list = [f'x = {x:+.3f}']
        case [x1, x2]:
            result_list = [f'x1 = {x1:+.3f}', f'x2 = {x2:+.3f}']
    for result in result_list:
        output_string += f'{result:^24}\n'
    output_string += f'\n{"Details":-^24}\n\n'
    details = equation.analyze()
    match details:
        case {'slope': slope, 'intercept': intercept}:
            details_list = [f'slope = {slope:>16.3f}', f'y-intercept = {intercept:>10.3f}']
        case {'x': x, 'y': y, 'min_max': min_max, 'concavity': concavity}:
            coord = f'({x:.3f}, {y:.3f})'
            details_list = [f'concavity = {concavity:>12}', f'{min_max} = {coord:>18}']
    for detail in details_list:
        output_string += f'{detail}\n'
    return output_string
lin_eq = LinearEquation(2, 3)
quadr_eq = QuadraticEquation(1, 2, 1)
print(solver(quadr_eq))


# In[1]:


import math

GRAVITATIONAL_ACCELERATION = 9.81
PROJECTILE = "∙"
x_axis_tick = "T"
y_axis_tick = "⊣"

class Projectile:
    __slots__ = ('__speed', '__height', '__angle')

    def __init__(self, speed, height, angle):
        self.__speed = speed
        self.__height = height
        self.__angle = math.radians(angle)
        
    def __str__(self):
        return f'''
Projectile details:
speed: {self.speed} m/s
height: {self.height} m
angle: {self.angle}°
displacement: {round(self.__calculate_displacement(), 1)} m
'''

    def __calculate_displacement(self):
        horizontal_component = self.__speed * math.cos(self.__angle)
        vertical_component = self.__speed * math.sin(self.__angle)
        squared_component = vertical_component**2
        gh_component = 2 * GRAVITATIONAL_ACCELERATION * self.__height
        sqrt_component = math.sqrt(squared_component + gh_component)
        
        return horizontal_component * (vertical_component + sqrt_component) / GRAVITATIONAL_ACCELERATION
        
    def __calculate_y_coordinate(self, x):
        height_component = self.__height
        angle_component = math.tan(self.__angle) * x
        acceleration_component = GRAVITATIONAL_ACCELERATION * x ** 2 / (
                2 * self.__speed ** 2 * math.cos(self.__angle) ** 2)
        y_coordinate = height_component + angle_component - acceleration_component

        return y_coordinate
    
    def calculate_all_coordinates(self):
        return [
            (x, self.__calculate_y_coordinate(x))
            for x in range(math.ceil(self.__calculate_displacement()))
        ]

    @property
    def height(self):
        return self.__height

    @property
    def angle(self):
        return round(math.degrees(self.__angle))

    @property
    def speed(self):
        return self.__speed

    @height.setter
    def height(self, n):
        self.__height = n

    @angle.setter
    def angle(self, n):
        self.__angle = math.radians(n)

    @speed.setter
    def speed(self, s):
       self.__speed = s
    
    def __repr__(self):
        return f'{self.__class__}({self.speed}, {self.height}, {self.angle})'

class Graph:
    __slots__ = ('__coordinates')

    def __init__(self, coord):
        self.__coordinates = coord

    def __repr__(self):
        return f"Graph({self.__coordinates})"

    def create_coordinates_table(self):
        table = '\n  x      y\n'
        for x, y in self.__coordinates:
            table += f'{x:>3}{y:>7.2f}\n'

        return table

    def create_trajectory(self):

        rounded_coords = [(round(x), round(y)) for x, y in self.__coordinates]

        x_max = max(rounded_coords, key=lambda i: i[0])[0]
        y_max = max(rounded_coords, key=lambda j: j[1])[1]

        matrix_list = [[" " for _ in range(x_max + 1)] for _ in range(y_max + 1)]

        for x, y in rounded_coords:
            matrix_list[-1 - y][x] = PROJECTILE

        matrix = ["".join(line) for line in matrix_list]

        matrix_axes = [y_axis_tick + row for row in matrix]
        matrix_axes.append(" " + x_axis_tick * (len(matrix[0])))

        graph = "\n" + "\n".join(matrix_axes) + "\n"

        return graph

def projectile_helper(speed,height,angle):
    #def projectile_helper(speed, height, angle):
    projectile = Projectile(speed, height, angle)
    coordinates = projectile.calculate_all_coordinates()
    graph = Graph(coordinates) 
    print(projectile)
    print(graph.create_coordinates_table())
    print(graph.create_trajectory())
        
projectile_helper(10,3,45)


# In[1]:


import copy
import random
from collections import defaultdict

# Consider using the modules imported above.

class Hat:
    def __init__(self, **kwargs: dict[str, int]) -> None:
        self.contents: list[str] = []
        for color, count in kwargs.items():
            [self.contents.append(color) for _ in range(count)]

    def __repr__(self) -> str:
        return str(self.contents)

    def draw(self,  n_balls: int):
        drawn = []
        if (n_balls >= len(self.contents)):
            drawn = self.contents
            self.contents = []
        else:
            for _ in range(n_balls):
                index = random.randrange(len(self.contents))
                drawn.append(self.contents[index])
                self.contents[index] = self.contents[-1]
                self.contents.pop()
        return drawn


def experiment(hat: Hat, expected_balls: dict[str, int], num_balls_drawn: int, num_experiments: int):
    n_desired_result = 0
    for _ in range(num_experiments):
        hat_copy = copy.deepcopy(hat)
        drawn = hat_copy.draw(num_balls_drawn)
        drawn_dict = defaultdict(lambda: 0)
        for ball in drawn:
            drawn_dict[ball] += 1
        desired_outcome = True
        for ball in expected_balls:
            if (drawn_dict[ball] < expected_balls[ball]):
                desired_outcome = False
                break
        if desired_outcome:
            n_desired_result += 1
    return n_desired_result/num_experiments


# In[1]:


get_ipython().system('pip install dash==2.8.1')


# In[1]:


get_ipython().system('pip install tensorflow')


# In[2]:


get_ipython().system('pip install scikit-learn')


# In[ ]:


get_ipython().system('pip uninstall tensorflow')


# In[ ]:


get_ipython().system('pip install tensorflow==2.11')


# In[ ]:




