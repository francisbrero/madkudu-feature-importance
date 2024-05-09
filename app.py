import streamlit as st
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import re

def extract_features(test_condition):
    # Improved regex to capture attribute names more cleanly, ignoring SQL and logical operators
    pattern = re.compile(r"\b([a-zA-Z_]+)__\b|\b([a-zA-Z_]+)\b")
    ignore_list = {
        "AND", "OR", "COALESCE", "NUMERIC", "IN", "AS", "unknown", "false", "BETWEEN", "LOWER", "NOT", "CAST", "VARCHAR", "com"
    }
    features = set()
    for match in pattern.finditer(test_condition):
        # Filter and add all captured feature names, avoiding adding None values and ignoring keywords
        features.update({group for group in match.groups() if group and group not in ignore_list})
    return features

def calculate_feature_importance(json_data):
    # Create node information including gini calculations
    nodes_info = {stat['node']: {'gini': 1 - 2 * stat['conversionRate'] * (1 - stat['conversionRate']), 
                                 'features': extract_features(stat['test'])} for stat in json_data['stats']}
    
    # Building parent-child relationships from the structure
    def build_tree(structure):
        node = structure['id']
        children = structure.get('children', [])
        for child in children:
            parent_child_map[child['id']] = node
            build_tree(child)
    
    parent_child_map = {}
    build_tree(json_data['structure'])

    # Calculate impurity reductions for features
    feature_impurity_reduction = defaultdict(float)
    for child_id, parent_id in parent_child_map.items():
        if parent_id in nodes_info and child_id in nodes_info:
            parent_gini = nodes_info[parent_id]['gini']
            child_gini = nodes_info[child_id]['gini']
            impurity_reduction = parent_gini - child_gini
            for feature in nodes_info[child_id]['features']:
                feature_impurity_reduction[feature] += impurity_reduction

    total_reduction = sum(feature_impurity_reduction.values())
    feature_importance = {feature: (reduction / total_reduction) for feature, reduction in feature_impurity_reduction.items()}
    return feature_importance

# Streamlit interface for the app
st.title('Decision Tree Feature Importance Calculator')
json_input = st.text_area("Paste JSON Here:", height=300)
if st.button('Calculate Feature Importance'):
    try:
        json_data = json.loads(json_input)
        importance = calculate_feature_importance(json_data)
        st.write("Feature Importance:", importance)
        
        # Plotting feature importance
        fig, ax = plt.subplots()
        ax.bar(importance.keys(), importance.values(), color='blue')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance Bar Chart')
        st.pyplot(fig)
    except json.JSONDecodeError:
        st.error("Invalid JSON")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add a sidebar with some additional information about the app such as the author, the purpose and the GitHub link
st.sidebar.title('About')
st.sidebar.info('This app calculates the feature importance of a decision tree model from a JSON input.')
# Add instructions on how to obtain the JSON
st.sidebar.markdown('''
#### Instructions:
1. Log into MadKudu
2. Open this url: https://studio.madkudu.com/api/tenant/3303/model/3/trees/1
3. Replace TENANT_NUMBER and MODEL_NUMBER with the ones you are interested in
4. Copy the JSON data from the request
''')
st.sidebar.title('Author')
st.sidebar.info('''
[Francis Brero](https://www.linkedin.com/in/francisbrero/)
|| [MadKudu](https://www.madkudu.com/)
|| [GitHub](https://github.com/francisbrero)
''')


