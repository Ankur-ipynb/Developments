# Databricks notebook source
# MAGIC %md
# MAGIC ### Read Data | DateFrame Reader API

# COMMAND ----------

dbutils.fs.ls('/FileStore/tables')

# COMMAND ----------

df = spark.read.format('csv').option('InferSchema',True).option('Header',True).load('/FileStore/tables/BigMart_Sales.csv')

# COMMAND ----------

df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Json Read

# COMMAND ----------

dbutils.fs.ls('/FileStore/tables')

# COMMAND ----------

df_json = spark.read.format('json').option('InferSchema',True)\
                                   .option('Header',True)\
                                   .option('multiline',False)\
                                       .load('/FileStore/tables/drivers.json')

# COMMAND ----------

df_json.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Schema Definition

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # DDL Schema

# COMMAND ----------

bigstore_ddl_schema = '''
                        Item_Identifier STRING,
                        Item_Weight STRING,
                        Item_Fat_Content STRING,
                        Item_Visibility DOUBLE,
                        Item_Type STRING,
                        Item_MRP DOUBLE,
                        Outlet_Identifier STRING,
                        Outlet_Established_Year INT,
                        Outlet_Size STRING,
                        Outlet_Location_Type STRING,
                        Outlet_Type STRING,
                        Item_Outlet_Sales DOUBLE
                     '''

# COMMAND ----------

df = spark.read.format('csv')\
                .schema(bigstore_ddl_schema)\
                .option('Header', True)\
                .load('/FileStore/tables/BigMart_Sales.csv')

# COMMAND ----------

df.display()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### StructType() Schema

# COMMAND ----------

from pyspark.sql.types import*
from pyspark.sql.functions import*

# COMMAND ----------

bigstore_struct_schema = StructType([
                                        StructField('Item_Identifier', StringType(), True),
                                        StructField('Item_Weight', StringType(), True),
                                        StructField('Item_Fat_Content', StringType(), True),
                                        StructField('Item_Visibility', StringType(), True),
                                        StructField('Item_Type', StringType(), True),
                                        StructField('Item_MRP', StringType(), True),
                                        StructField('Outlet_Identifier', StringType(), True),
                                        StructField('Outlet_Established_Year', StringType(),True),
                                        StructField('Outlet_Size', StringType(), True),
                                        StructField('Outlet_Location_Type', StringType(), True),
                                        StructField('Outlet_Type', StringType(), True),
                                        StructField('Item_Outlet_Sales', StringType(), True)
])

# COMMAND ----------

df = spark.read.format('csv')\
            .schema(bigstore_struct_schema)\
            .option('header', True)\
            .load('/FileStore/tables/BigMart_Sales.csv')

# COMMAND ----------

 df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Back to Normal DF

# COMMAND ----------

df = spark.read.format('csv')\
        .option('Inferschema', True)\
        .option('Header', True)\
        .load('/FileStore/tables/BigMart_Sales.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC ### SELECT

# COMMAND ----------

df.display()

# COMMAND ----------

df.select('Item_Identifier', 'Item_Weight','Item_Fat_Content').display()

# COMMAND ----------

df.select(col('Item_Visibility'), col('Item_Type'), col('Item_MRP')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC Use col() for better segregation

# COMMAND ----------

# MAGIC %md
# MAGIC ### ALIAS

# COMMAND ----------

df.select(col('Item_Identifier').alias('Item_ID')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### FILTER

# COMMAND ----------

# MAGIC %md
# MAGIC #### Scenario 01

# COMMAND ----------

df.filter(col('Item_Fat_Content')=='Regular').display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Scenario 02

# COMMAND ----------

df.filter( (col('Item_Type')=='Soft Drinks' ) & (col('Item_Weight')<10) ).display() 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Scenario 03

# COMMAND ----------

df.filter((col('Outlet_Size').isNull()) & (col('Outlet_Location_Type').isin('Tier 1','Tier 2'))).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### withColumnRenamed | rename at Data Frame level

# COMMAND ----------

df.withColumnRenamed('Item_Weight', 'Item_Wt').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### withColumn | new and modify col

# COMMAND ----------

# MAGIC %md
# MAGIC #### New Column

# COMMAND ----------

df = df.withColumn('Flag', lit('new'))

# COMMAND ----------

df.display()

# COMMAND ----------

df.withColumn('multiply', col('Item_Weight')*col('Item_MRP')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Modify Existing Col

# COMMAND ----------

df.withColumn('Item_Fat_Content', regexp_replace(col('Item_Fat_content'), 'Regular','Reg'))\
    .withColumn('Item_Fat_Content', regexp_replace(col('Item_Fat_Content'), 'Low Fat','Lf')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Type Casting

# COMMAND ----------

df = df.withColumn('Item_Weight', col('Item_Weight').cast(StringType()))

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### SORT

# COMMAND ----------

df.sort(col('Item_Weight').desc()).display()

# COMMAND ----------

df.sort(col('Item_Visibility').asc()).display()

# COMMAND ----------

df.sort(['Item_Weight','Item_Visibility'], ascending = ['0,0']).display()

# COMMAND ----------

df.sort(['Item_Weight','Item_Visibility'], ascending = [0,1]).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### LIMIT

# COMMAND ----------

df.limit(10).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### DROP

# COMMAND ----------

df.drop('Item_Visibility','Item_Weight').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### DROP_DUPLICATES | D-DUP

# COMMAND ----------

df.dropDuplicates().display()

# COMMAND ----------

df.drop_duplicates(subset=['Item_Type']).display()

# COMMAND ----------

df.distinct().display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### UNION and UNION BY NAME

# COMMAND ----------



# COMMAND ----------

data1 = [('1', 'Py'),
         ('2', 'Spark')]
schema1 = 'id STRING', 'name STRING'
df1 = spark.createDataFrame(data1, schema1)
data2 = [('Sample','3'),
         ('DataFrame','4')]
schema2 = 'name STRING', 'id STRING'
df2 = spark.createDataFrame(data2, schema2)

# COMMAND ----------

df1.display()

# COMMAND ----------

df2.display()

# COMMAND ----------

df1.union(df2).display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### union by name

# COMMAND ----------

df1.unionByName(df2).display()


# COMMAND ----------

# MAGIC %md
# MAGIC ### String Functions

# COMMAND ----------

import pyspark.sql.functions

# COMMAND ----------

df.select(initcap('Item_Type')).display()

# COMMAND ----------

df.select(lower('Item_Type')).display()

# COMMAND ----------

df.select(upper('Item_Type').alias('Upper_Item_Type')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Date Functions

# COMMAND ----------

df = spark.read.format('csv')\
    .option('InferSchema', True)\
    .option('Header', True)\
    .load('/FileStore/tables/BigMart_Sales.csv')

# COMMAND ----------

df = df.withColumn('curr_date', current_date())

# COMMAND ----------

df = df.withColumn('week_after', date_add('curr_date',7))
df.limit(3).display()

# COMMAND ----------

df = df.withColumn('week_before', date_sub('curr_date', 7))
df.limit(3).display()

# COMMAND ----------

df = df.withColumn('week_before', date_add('curr_date', -7))
df.limit(3).display(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Date Diff

# COMMAND ----------

df = df.withColumn('datediff', datediff('week_after','curr_date'))
df.limit(3).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Date_Format

# COMMAND ----------

df = df.withColumn('week_after',date_format('week_after','dd-MM-yyyy'))
df.limit(3).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### NULL Handling

# COMMAND ----------

df.dropna('all').display()

# COMMAND ----------

df.dropna('any').display()

# COMMAND ----------

df.dropna(subset=['Outlet_Size']).display()

# COMMAND ----------

df.fillna('NotAvailable', subset=['outlet_Size']).display()

# COMMAND ----------

df.fillna('NotAvailable').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### SPLIT and Indexing

# COMMAND ----------

df.withColumn('Outlet_Type', split('Outlet_Type', ' ')).display()

# COMMAND ----------

df.withColumn('OutLet_Type', split('Outlet_type',' ')[1]).limit(3).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### EXPLODE

# COMMAND ----------

df_exp = df.withColumn('Outlet_Type', split('Outlet_Type',' '))
df_exp.limit(3).display()

# COMMAND ----------

df_exp.withColumn('Outlet_Type',explode('Outlet_Type')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Array Contains

# COMMAND ----------

df_exp.withColumn('Type1_flag', array_contains('Outlet_Type', 'Type1')).limit(3).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### GROUP_BY

# COMMAND ----------

df.groupBy('Item_Type').agg(sum('Item_MRP')).display()

# COMMAND ----------

df.groupBy('Item_Type').agg(avg('Item_MRP')).display()

# COMMAND ----------

df.groupBy('Item_Type','Outlet_Size').agg(sum('Item_MRP').alias('Total_MRP')).display()

# COMMAND ----------

df.groupBy('Item_Type','Outlet_Size').agg(sum('Item_MRP'),avg('Item_MRP')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Collect_List

# COMMAND ----------

book_data = [('user1','book1'),
             ('user1','book2'),
             ('user2','book2'),
             ('user2','book4'),
             ('user3','book1')]
schema = 'user string, book string'
df_book = spark.createDataFrame(book_data, schema)
df_book.display()

# COMMAND ----------

df_book.groupBy('user').agg(collect_list('book')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### PIVOT

# COMMAND ----------

df.groupBy('Item_Type').pivot('Outlet_Size').agg(avg('Item_MRP')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC > ### When-OtherWise

# COMMAND ----------

df = df.withColumn('veg_flag', when(col('Item_Type')=='Meat','Non-Veg').otherwise('Veg'))
df.limit(5).display()

# COMMAND ----------

df.withColumn('veg_exp_flag', when(((col('veg_flag')=='Veg')&(col('Item_MRP')<100)),'Veg_Inexpensive')\
                            .when((col('veg_flag')=='Veg')&(col('Item_MRP')>100),'Veg_Expensive')\
                            .otherwise('Non_Veg')).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### JOINS

# COMMAND ----------

data_j1 = [('1','gaur','d01'),
           ('2','kit','d02'),
           ('3','sam','d03'),
           ('4','tim','d03'),
           ('5','aman','d05'),
           ('6','nad','d06')]
schema_j1='emp_id STRING, emp_name STRING, dept_id STRING'
df_j1 = spark.createDataFrame(data_j1, schema_j1)
data_j2 = [('d01','HR'),
           ('d02','Marketing'),
           ('d03','Account'),
           ('d04','IT'),
           ('d05','Finance')]
schema_j2 = 'dept_id STRING, dept STRING'
df_j2=spark.createDataFrame(data_j2,schema_j2)

# COMMAND ----------

df_j1.display()

# COMMAND ----------

df_j2.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### inner join

# COMMAND ----------

df_j1.join(df_j2, df_j1['dept_id']==df_j2['dept_id'],'inner').display()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Left Join

# COMMAND ----------

df_j1.join(df_j2, df_j1['dept_id']==df_j2['dept_id'],'left').display()

# COMMAND ----------

df_j1.join(df_j2, df_j1['dept_id']==df_j2['dept_id'],'right').display()

# COMMAND ----------

df_j1.join(df_j2, df_j1['dept_id']==df_j2['dept_id'],'anti').display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### WINDOW FUNCTIONS

# COMMAND ----------

from pyspark.sql.window import Window

# COMMAND ----------

df.withColumn('rowCol', row_number().over(Window.orderBy('Item_Identifier').desc())).limit(3).display()

# COMMAND ----------

df.withColumn('rank',rank().over(Window.orderBy(col('Item_Identifier').desc()))).limit(10).display()

# COMMAND ----------

df.withColumn('denserank',dense_rank().over(Window.orderBy(col('Item_Identifier').desc()))).limit(10).display()
        

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cumlative Sum from current to preceeding row

# COMMAND ----------

df.withColumn('cumsum', sum('Item_MRP').over(Window.orderBy('Item_Type').rowsBetween(Window.unboundedPreceding, Window.currentRow))).limit(5).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### User Defined Functions | UDF

# COMMAND ----------

def bigstore_func(x):
    return x*x

# COMMAND ----------

bigstore_udf = udf(bigstore_func)

# COMMAND ----------

df.withColumn('squared_MRP', bigstore_udf('Item_MRP')).limit(3).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### DATA WRITING

# COMMAND ----------

df.write.format('csv')\
    .save('/FileStore/tables/CSV/data.csv')

# COMMAND ----------

df.write.format('csv')\
    .mode('append')\
    .save('/FileStore/tables/CSV/data.csv')

# COMMAND ----------

df.write.format('csv')\
    .mode('overwrite')\
    .option('path','/FileStore/tables/CSV/data.csv')\
    .save()

# COMMAND ----------

df.write.format('csv')\
    .mode('error')\
    .option('path','/FileStore/tables/CSV/data.csv')\
    .save()

# COMMAND ----------

df.write.format('csv')\
    .mode('ignore')\
    .option('path','/FileStore/tables/CSV/data.csv')\
    .save()

# COMMAND ----------

# MAGIC %md
# MAGIC ### PARQUET FILE FORMAT | Columnar Format

# COMMAND ----------

df.write.format('parquet')\
    .mode('overwrite')\
    .option('path','/FileStore/tables/CSV/data.csv')\
    .save()

# COMMAND ----------

# MAGIC %md
# MAGIC #### create table

# COMMAND ----------

df.write.format('csv')\
    .mode('overwrite')\
    .saveAsTable('BigStore_table')

# COMMAND ----------

# MAGIC %md
# MAGIC Managed vs External Tables

# COMMAND ----------

df.limit(5).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Spark SQL

# COMMAND ----------

df.createTempView('BigStore_View')

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from BigStore_View where Item_Fat_Content = 'Low Fat'

# COMMAND ----------

df_sql = spark.sql("select * from BigStore_View where Item_Fat_Content = 'Low Fat'")
df_sql.limit(5).display()

# COMMAND ----------


