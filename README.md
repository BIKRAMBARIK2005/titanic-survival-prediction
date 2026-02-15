# 🚢 Titanic Survival Prediction – Day 29 Project

This project predicts whether a passenger would survive the Titanic disaster using Machine Learning.

It includes:

- ✅ Model built **without Pipeline**
- ✅ Model built **with Pipeline (Recommended)**
- ✅ Streamlit Web App
- ✅ Clean folder structure
- ✅ Deployment-ready setup

---

## 📊 Project Overview

The Titanic dataset is used to train a classification model that predicts passenger survival based on:

- Passenger Class (Pclass)
- Sex
- Age
- Siblings/Spouses aboard (SibSp)
- Parents/Children aboard (Parch)
- Fare
- Embarked Port

Two different approaches are implemented:

---

## 🔹 1️⃣ Without Pipeline Model

- Manual encoding (OneHotEncoder)
- Separate preprocessing and model
- Multiple `.pkl` files:
  - `clf.pkl`
  - `ohe_sex.pkl`
  - `ohe_embarked.pkl`

This version demonstrates understanding of preprocessing steps individually.

---

## 🔹 2️⃣ Pipeline Model (Recommended)

- Uses `sklearn Pipeline`
- Combines preprocessing + model
- Single file:
  - `pipe.pkl`

This version is cleaner, production-ready, and easier to deploy.

---

## 🌐 Streamlit Web App

The project includes a fully responsive web application built using **Streamlit**.

Users can:

- Select model type (Pipeline / Without Pipeline)
- Enter passenger details
- Instantly get survival prediction

---

## 📂 Folder Structure

