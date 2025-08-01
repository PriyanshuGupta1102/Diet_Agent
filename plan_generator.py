import os
import json
from dotenv import load_dotenv
from typing import Optional

# Pydantic is used to define the exact output data structure
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# --- 1. Define the Desired JSON Structure using Pydantic ---
class Meal(BaseModel):
    name: str = Field(description="Name of the meal")
    time: str = Field(description="Suggested time for the meal, e.g., '08:00'")
    calories: int = Field(description="Estimated calories for the meal")
    notes: Optional[str] = Field(description="A brief note about the meal's benefits or ingredients")

class Meals(BaseModel):
    breakfast: Meal
    lunch: Meal
    snack: Meal
    dinner: Meal

class Macronutrients(BaseModel):
    protein_grams: int = Field(description="Total grams of protein for the day")
    carbs_grams: int = Field(description="Total grams of carbohydrates for the day")
    fat_grams: int = Field(description="Total grams of fat for the day")

class DietPlan(BaseModel):
    daily_calories: int = Field(description="The target total daily calorie intake for the plan")
    macronutrients: Macronutrients = Field(description="The target macronutrient breakdown for the day")
    meals: Meals = Field(description="A dictionary containing details for breakfast, lunch, snack, and dinner")

# --- 2. Set up the Model and Prompt ---
def generate_diet_plan(user_input: str) -> DietPlan:
    """
    Generates a structured diet plan object based on user input.
    This function RETURNS the Pydantic object, it does not print it.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
    llm_with_plan_schema = llm.with_structured_output(DietPlan)
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an expert nutritionist. Your task is to create a detailed, one-day diet plan based on the user's information. "
         "You MUST generate the output in the required JSON format. Do not produce any other text or explanation outside of the JSON structure."),
        ("human", "{user_input}")
    ])
    chain = prompt | llm_with_plan_schema
    
    plan = chain.invoke({"user_input": user_input})
    
    # Return the Pydantic object
    return plan

def main():
    """
    This main function is for demonstration when running the script directly.
    It shows how to call the generator and receive the object.
    """
    print("Demonstration: Calling the diet plan generator...")
    
    user_details = """
    I need a diet plan with the following details:
    - Age: 30
    - Gender: Male
    - Weight: 85 kg
    - Height: 180 cm
    - Goal: Weight Loss (mild deficit)
    - Dietary Restrictions: None
    - Activity Level: Moderately Active (exercise 3-4 times a week)
    """
    
    # Generate the structured plan and store the returned object in a variable
    generated_plan_object = generate_diet_plan(user_input=user_details)
    
    print("...Diet plan object has been successfully generated and is stored in a variable.")
    print("In a real application, this object would now be sent to an API, saved to a database, etc.")
    
    # For proof, we can print just one small part of the object
    print(f"\nExample of accessing data from the returned object:")
    print(f"Breakfast Name: {generated_plan_object.meals.breakfast.name}")


if __name__ == "__main__":
    main()