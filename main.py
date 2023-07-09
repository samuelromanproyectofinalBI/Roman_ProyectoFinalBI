import streamlit as st
from multiapp import MultiApp
from deploy import modeloSVC, modeloRFS, modeloArbolDecision

app = MultiApp()
st.markdown("# Inteligencia de Negocios - Equipo E - Trabajo Final Individual")
st.markdown("## Docente: Ernesto Cancho Rodriguez")
st.markdown("## Estudiante: Samuel Aaron Roman Cespedes")

app.add_app("Modelo SVC", modeloSVC.app)
app.add_app("Modelo Árbol de decisión", modeloArbolDecision.app)
app.add_app("Modelo Random Forest Regression", modeloRFS.app)
app.run()