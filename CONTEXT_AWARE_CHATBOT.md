# Context-Aware AI Assistant - Implementation Guide

## Overview

The chatbot has been transformed into a **context-aware AI assistant** that automatically adapts its responses based on what the user is currently viewing on screen. It follows ethical guidelines and provides supportive, actionable recommendations.

## Core Behavior Rules

### 1. Probabilistic Language
- All risk outputs are framed as **probabilistic risk**, never certainty
- Uses phrases like "probabilistic risk", "likelihood", "suggests"

### 2. Ethical Guidelines
- **Never recommends:** punitive actions, disciplinary measures, coercion, termination, pressure tactics, or surveillance
- **Always frames recommendations as:** supportive, ethical retention interventions

### 3. Response Structure
Unless the user asks otherwise, responses follow this structure:
1. **What the data indicates** - Clear explanation of the situation
2. **Why it matters now** - Urgency or trend context
3. **What to do next** - Maximum 2-3 concrete, actionable steps

### 4. Context Adaptation
- **Dashboard View:** Focuses on organizational trends, departments, prioritization
- **Employee Profile View:** Focuses on individual employee, risk drivers, manager actions

## How It Works

### Context Detection

The assistant automatically detects:
- **Current page** (Dashboard, Employee Profile, Department Analysis, Search)
- **Selected employee** (when on Employee Profile page)
- **Selected department** (when on Department Analysis page)

### Response Adaptation

**On Dashboard:**
- Answers at HR/leadership level
- Focuses on trends and patterns
- Provides organizational insights
- Avoids individual employee speculation

**On Employee Profile:**
- Focuses only on the current employee
- Explains risk score and drivers
- Recommends manager-executable actions
- Suggests conversation approaches

## Example Interactions

### Dashboard Context

**User:** "How many employees are at high risk?"
**Assistant:** 
```
What the data indicates:
There are 56 employees (11.2%) at high or critical risk (60%+ risk score).

Why it matters now:
This represents a significant portion of the workforce requiring proactive 
intervention. Early action can prevent further escalation.

What to do next:
1. Review the high-risk employee list to identify patterns
2. Prioritize by department or role criticality
3. Develop department-level retention strategies
```

### Employee Profile Context

**User:** "What should I do?"
**Assistant:**
```
What the data indicates:
John Smith has a 67.3% probabilistic risk of voluntary departure (High risk level).

Why it matters now:
This warrants proactive intervention. The upward trend suggests risk factors 
are intensifying.

What to do next:
1. Proactive check-in with manager to discuss career goals and satisfaction
2. Address primary risk drivers identified in their profile
3. Monitor engagement and provide regular feedback
```

**User:** "Why are they at risk?"
**Assistant:**
```
What the data indicates:
The primary factors contributing to John Smith's risk are:

1. No Career Progression (42% influence)
   No promotion in 18 months

2. Below-Market Compensation (31% influence)
   Salary 12% below market average

Why it matters now:
These factors, when combined, create a pattern that suggests potential 
dissatisfaction. Addressing the top driver alone could significantly reduce 
overall risk.

What to do next:
1. Discuss career path and create a development plan
2. Identify growth opportunities within the organization
3. Set clear milestones for advancement
```

## Supported Query Types

### Employee Profile Context
- "What's their risk score?"
- "Why are they at risk?"
- "What should I do?"
- "What's the trend?"
- "How should I approach a conversation?"
- "What are the risk drivers?"

### Dashboard Context
- "How many employees are at high risk?"
- "Which department has the highest risk?"
- "Show me risk statistics"
- "What should we prioritize?"
- "Find [Employee Name]"
- "Compare departments"

## Tone & Style

- **Professional and calm** - Trusted advisor, not alarm system
- **Supportive** - Focuses on helping, not warning
- **Concise** - Practical and decision-supportive
- **Plain language** - Avoids jargon unless requested
- **Actionable** - Provides concrete next steps

## Technical Implementation

### Context Passing

The UI context is automatically built and passed to the chatbot:

```python
ui_context = {
    'view_type': 'employee_profile',  # or 'dashboard', 'department_analysis', 'search'
    'employee_id': 'EMP001',
    'employee_name': 'John Smith'
}

response = chatbot.process_query(user_query, ui_context)
```

### Response Format

All responses follow a consistent structure:

```python
{
    'type': 'text',  # or 'employee_detail', 'employee_list'
    'content': 'Formatted response text',
    'data': {...},  # Optional: supporting data
    'suggestions': [...]  # Optional: follow-up suggestions
}
```

## Benefits

1. **Context-Aware:** Understands what user is viewing
2. **Ethical:** Follows supportive, non-punitive guidelines
3. **Actionable:** Provides concrete next steps
4. **Adaptive:** Changes tone and focus based on context
5. **Professional:** Maintains appropriate level of detail

## Usage Tips

- Ask questions naturally - the assistant understands context
- On Employee Profile: Ask about the current employee
- On Dashboard: Ask about organizational trends
- Be specific for better results
- Use the suggestions for follow-up questions

---

**The context-aware AI assistant is now live and ready to help!**

