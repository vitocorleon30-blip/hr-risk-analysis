"""
Context-Aware AI Assistant for HR Risk Analysis Dashboard
==========================================================
Intelligent assistant that adapts responses based on current UI context.
Enhanced with chart generation, pattern analysis, and recommendation capabilities.
"""

import pandas as pd
import re
from typing import Dict, List, Any, Optional, Tuple
from rapidfuzz import fuzz, process
import plotly.graph_objects as go
from predict import (
    predict_attrition_risk,
    get_employee_by_id,
    get_high_risk_employees,
    get_risk_summary,
    get_department_summary
)


class HRChatbot:
    """
    Context-aware AI assistant for HR Risk Analysis.
    Adapts responses based on current UI view (dashboard vs employee profile).
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.conversation_history = []
        self.last_response_type = None
        self.last_topic = None
        self._build_search_index()
        
    def _build_search_index(self):
        """Build search index for faster lookups."""
        self.employee_names = self.df['Name'].tolist()
        self.employee_ids = self.df['Employee_ID'].tolist()
        self.departments = sorted(self.df['Department'].unique().tolist())
        
    def process_query(self, query: str, ui_context: Dict[str, Any] = None, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Process a natural language query with UI context awareness and conversation history.
        
        Args:
            query: User's natural language query
            ui_context: Current UI context including view_type, selected employee, etc.
            conversation_history: List of previous messages in the conversation
            
        Returns:
            Dictionary with response type, content, and optional data
        """
        if not query or not query.strip():
            return self._get_contextual_help(ui_context)
        
        query_lower = query.lower().strip()
        ui_context = ui_context or {}
        conversation_history = conversation_history or []
        view_type = ui_context.get('view_type', 'dashboard')
        
        # Check for follow-up questions (but not if it's a clear new topic)
        is_clear_new_topic = any(word in query_lower for word in [
            'show me', 'create', 'generate', 'find', 'search', 'which department',
            'how many', 'what is', 'tell me about'
        ])
        
        if conversation_history and not is_clear_new_topic and self._is_followup_question(query_lower, conversation_history):
            return self._handle_followup(query, query_lower, conversation_history, ui_context)
        
        # Context-aware processing
        if view_type == 'employee_profile':
            response = self._handle_employee_context_query(query, ui_context)
        elif view_type == 'chatbot':
            # Enhanced chatbot mode with all capabilities
            response = self._handle_dashboard_context_query(query, ui_context, conversation_history)
        else:
            response = self._handle_dashboard_context_query(query, ui_context, conversation_history)
        
        # Store response context for follow-up questions
        self.last_response_type = response.get('type', 'text')
        if 'data' in response:
            if 'employee_name' in response.get('data', {}):
                self.last_topic = response['data'].get('employee_name')
            elif 'department' in str(response.get('data', {})):
                self.last_topic = 'department'
        
        return response
    
    def _is_followup_question(self, query_lower: str, conversation_history: List[Dict]) -> bool:
        """Check if the query is a follow-up question."""
        # Strong follow-up indicators
        strong_followup = [
            'tell me more', 'more about', 'what about', 'how about',
            'what else', 'anything else', 'and then', 'also',
            'explain more', 'elaborate', 'more details', 'more information',
            'yes', 'no', 'ok', 'okay', 'sure', 'thanks', 'thank you',
            'that', 'this', 'them', 'it', 'they'  # Pronouns
        ]
        
        # Check if query is very short (likely a follow-up)
        if len(query_lower.split()) <= 2:
            return True
        
        # Check for strong follow-up indicators
        if any(indicator in query_lower for indicator in strong_followup):
            return True
        
        # Check if query starts with question words but is short (likely follow-up)
        question_words = ['why', 'how', 'what', 'when', 'where']
        if any(query_lower.startswith(word) for word in question_words) and len(query_lower.split()) <= 5:
            return True
        
        return False
    
    def _handle_followup(self, query: str, query_lower: str, conversation_history: List[Dict], context: Dict) -> Dict[str, Any]:
        """Handle follow-up questions based on conversation history."""
        # Get last assistant response
        last_assistant_msg = None
        for msg in reversed(conversation_history):
            if msg.get('role') == 'assistant':
                last_assistant_msg = msg
                break
        
        if not last_assistant_msg:
            return self._handle_dashboard_context_query(query, context, conversation_history)
        
        last_content = last_assistant_msg.get('content', '').lower()
        
        # Handle "tell me more" type questions
        if any(word in query_lower for word in ['tell me more', 'more', 'details', 'elaborate', 'explain more', 'expand']):
            if 'risk' in last_content or 'employee' in last_content:
                response = self._handle_risk_overview(query, context, conversation_history)
                # Add conversational intro
                response['content'] = "Sure! " + response['content']
                return response
            elif 'department' in last_content:
                response = self._handle_department_analysis(query, context, conversation_history)
                response['content'] = "Of course! " + response['content']
                return response
            elif 'recommend' in last_content or 'action' in last_content:
                response = self._handle_recommendations(query, context, conversation_history)
                response['content'] = "Here are more details:\n\n" + response['content']
                return response
            elif 'chart' in last_content or 'visual' in last_content:
                return self._handle_chart_request(query, context, conversation_history)
            else:
                # Generic "tell me more" - provide additional context
                return {
                    'type': 'text',
                    'content': f"Based on what we discussed, here's additional information:\n\n" +
                              self._handle_statistics(query, context, conversation_history).get('content', '')
                }
        
        # Handle "what about X" questions
        if 'what about' in query_lower or 'how about' in query_lower:
            # Extract the topic
            if 'department' in query_lower:
                return self._handle_department_analysis(query, context, conversation_history)
            elif 'employee' in query_lower or any(name.lower() in query_lower for name in self.employee_names[:10]):
                return self._handle_employee_search(query, context, conversation_history)
            elif 'risk' in query_lower:
                return self._handle_risk_overview(query, context, conversation_history)
        
        # Handle "and" / "also" questions
        if any(word in query_lower for word in ['and', 'also', 'additionally', 'what else']):
            if 'statistics' in last_content or 'overview' in last_content:
                return self._handle_statistics(query, context, conversation_history)
            elif 'department' in last_content:
                return self._handle_comparison(query, context, conversation_history)
            elif 'risk' in last_content:
                return self._handle_pattern_analysis(query, context, conversation_history)
        
        # Handle "why" questions
        if query_lower.startswith('why'):
            if 'risk' in last_content:
                return self._handle_pattern_analysis(query, context, conversation_history)
            else:
                stats_response = self._handle_statistics(query, context, conversation_history)
                return {
                    'type': 'text',
                    'content': f"Based on our previous discussion, here's more context:\n\n" + 
                              stats_response.get('content', '')
                }
        
        # Handle "how" questions
        if query_lower.startswith('how'):
            if 'many' in query_lower:
                return self._handle_statistics(query, context, conversation_history)
            elif 'compare' in query_lower or 'different' in query_lower:
                return self._handle_comparison(query, context, conversation_history)
            else:
                return self._handle_recommendations(query, context, conversation_history)
        
        # Default: try to understand from context
        return self._handle_dashboard_context_query(query, context, conversation_history)
    
    def _handle_employee_context_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle queries in employee profile context."""
        query_lower = query.lower()
        employee_id = context.get('employee_id')
        employee_name = context.get('employee_name')
        
        if not employee_id:
            return {
                'type': 'text',
                'content': "I need to know which employee you're viewing. Please select an employee first."
            }
        
        # Get employee data
        employee_row = self.df[self.df['Employee_ID'] == employee_id]
        if employee_row.empty:
            return {
                'type': 'text',
                'content': f"Could not find employee {employee_id}."
            }
        
        prediction = predict_attrition_risk(employee_row.iloc[0])
        risk_score = prediction.get('risk_score', 0)
        risk_level = prediction.get('risk_level', 'N/A')
        drivers = prediction.get('top_3_drivers', [])
        
        # Interpret query intent
        if any(word in query_lower for word in ['risk', 'score', 'level', 'at risk']):
            return self._explain_employee_risk(prediction, context)
        
        if any(word in query_lower for word in ['why', 'reason', 'driver', 'factor', 'cause']):
            return self._explain_risk_drivers(prediction, context)
        
        if any(word in query_lower for word in ['what', 'do', 'action', 'recommend', 'suggest', 'help', 'intervention']):
            return self._recommend_actions(prediction, context)
        
        if any(word in query_lower for word in ['trend', 'change', 'increasing', 'decreasing', 'stable']):
            return self._explain_trend(prediction, context)
        
        if any(word in query_lower for word in ['manager', 'talk', 'conversation', 'discuss', 'meeting']):
            return self._suggest_conversation_approach(prediction, context)
        
        # Default: Provide risk overview
        return self._explain_employee_risk(prediction, context)
    
    def _handle_dashboard_context_query(self, query: str, context: Dict[str, Any], conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Handle queries in dashboard context."""
        query_lower = query.lower()
        
        # FIRST: Handle conversational/greeting queries BEFORE HR queries
        conversational_keywords = [
            'how are you', 'how\'s it going', 'what\'s up', 'how do you do',
            'hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening',
            'thanks', 'thank you', 'appreciate', 'goodbye', 'bye', 'see you',
            'ok', 'okay', 'sure', 'alright', 'fine', 'good', 'great', 'nice',
            'yes', 'no', 'maybe', 'perhaps', 'i see', 'got it', 'understood'
        ]
        
        # Check for pure conversational queries (not mixed with HR terms)
        is_pure_conversational = any(phrase in query_lower for phrase in conversational_keywords)
        has_hr_keywords = any(word in query_lower for word in [
            'risk', 'employee', 'department', 'statistics', 'chart', 'recommend',
            'analyze', 'compare', 'find', 'search', 'show me', 'how many'
        ])
        
        if is_pure_conversational and not has_hr_keywords:
            return {
                'type': 'text',
                'content': self._handle_greeting(query_lower, conversation_history)
            }
        
        # Chart generation queries
        if any(word in query_lower for word in ['chart', 'graph', 'visualize', 'plot', 'show me a', 'create a']):
            return self._handle_chart_request(query, context, conversation_history)
        
        # Pattern analysis queries
        if any(word in query_lower for word in ['pattern', 'trend', 'correlation', 'analyze', 'insight']):
            return self._handle_pattern_analysis(query, context, conversation_history)
        
        # Recommendation queries
        if any(word in query_lower for word in ['recommend', 'recommendation', 'suggest', 'what should', 'propose', 'action']):
            return self._handle_recommendations(query, context, conversation_history)
        
        # Check conversation history for context
        last_user_query = None
        if conversation_history:
            for msg in reversed(conversation_history):
                if msg.get('role') == 'user':
                    last_user_query = msg.get('content', '').lower()
                    break
        
        # Risk-related queries (check first as it's most common)
        if any(word in query_lower for word in ['high risk', 'critical', 'at risk', 'risk']):
            response = self._handle_risk_overview(query, context)
            # Make response more conversational if it's a follow-up
            if last_user_query and 'risk' in last_user_query:
                response['content'] = self._make_more_conversational(response['content'], query)
            return response
        
        # Department queries
        if any(word in query_lower for word in ['department', 'dept', 'team', 'division']):
            response = self._handle_department_analysis(query, context)
            if last_user_query and 'department' in last_user_query:
                response['content'] = self._make_more_conversational(response['content'], query)
            return response
        
        # Statistics queries (but NOT "how are you" - that's conversational)
        if any(word in query_lower for word in ['how many', 'total', 'statistics', 'summary', 'overview']) and 'how are you' not in query_lower:
            response = self._handle_statistics(query, context)
            if last_user_query:
                response['content'] = self._make_more_conversational(response['content'], query)
            return response
        
        # "Show me" queries (but check it's not conversational)
        if 'show me' in query_lower and not any(word in query_lower for word in ['how are you', 'hello', 'hi', 'hey']):
            response = self._handle_statistics(query, context)
            if last_user_query:
                response['content'] = self._make_more_conversational(response['content'], query)
            return response
        
        # Employee search
        if any(word in query_lower for word in ['find', 'search', 'employee', 'who is', 'tell me about']):
            return self._handle_employee_search(query, context, conversation_history)
        
        # Comparison queries
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference', 'versus']):
            return self._handle_comparison(query, context, conversation_history)
        
        # Priority/focus queries
        if any(word in query_lower for word in ['priority', 'focus', 'urgent', 'important', 'next', 'should i']):
            return self._handle_prioritization(query, context, conversation_history)
        
        # Question words - try to understand intent better
        if query_lower.startswith(('what', 'which', 'who', 'where', 'when')):
            # More specific handling
            if 'department' in query_lower:
                return self._handle_department_analysis(query, context)
            elif 'employee' in query_lower:
                return self._handle_employee_search(query, context)
            elif 'risk' in query_lower:
                return self._handle_risk_overview(query, context)
            else:
        return self._handle_statistics(query, context)
        
        # Default: If it's unclear, provide helpful response
        return {
            'type': 'text',
            'content': "I'm here to help with HR analytics! I can help you with:\n\n• Employee risk analysis\n• Department comparisons\n• Data visualizations\n• Strategic recommendations\n• Pattern analysis\n\nWhat would you like to know?"
        }
    
    def _make_more_conversational(self, content: str, current_query: str) -> str:
        """Make responses more conversational and less repetitive."""
        # Remove repetitive headers if it's a follow-up
        if len(current_query.split()) <= 5:  # Short query = likely follow-up
            # Remove "What the data indicates:" header for shorter responses
            if "**What the data indicates:**" in content:
                content = content.replace("**What the data indicates:**\n\n", "")
            # Make it more direct
            if content.startswith("**"):
                # Find first newline after header
                lines = content.split('\n')
                if len(lines) > 2:
                    content = '\n'.join(lines[2:])  # Skip header lines
        
        return content
    
    def _handle_greeting(self, query_lower: str, conversation_history: List[Dict]) -> str:
        """Handle greetings and acknowledgments naturally."""
        if 'how are you' in query_lower or 'how\'s it going' in query_lower or 'how do you do' in query_lower:
            return "I'm doing well, thank you for asking! I'm here and ready to help you with HR analytics. What would you like to explore today?"
        elif 'thanks' in query_lower or 'thank you' in query_lower:
            return "You're welcome! Is there anything else I can help you with regarding employee risk analysis or HR insights?"
        elif 'hello' in query_lower or 'hi' in query_lower or 'hey' in query_lower:
            return "Hello! I'm your HR Analytics Assistant. I can help you analyze employee risk data, compare departments, create visualizations, and provide recommendations. What would you like to know?"
        elif 'goodbye' in query_lower or 'bye' in query_lower or 'see you' in query_lower:
            return "Goodbye! Feel free to come back anytime you need help with HR analytics."
        elif 'ok' in query_lower or 'okay' in query_lower or 'sure' in query_lower or 'alright' in query_lower:
            return "Great! What would you like to explore? I can help with risk analysis, department comparisons, or generate visualizations."
        elif query_lower.strip() in ['yes', 'no'] and len(query_lower.split()) <= 2:
            if conversation_history:
                return "Got it! What else can I help you with?"
            else:
                return "How can I assist you with your HR risk analysis today?"
        else:
            return "I'm here to help! What would you like to know about your workforce?"
    
    def _explain_employee_risk(self, prediction: Dict, context: Dict) -> Dict[str, Any]:
        """Explain employee risk score in context-aware, supportive manner."""
        risk_score = prediction.get('risk_score', 0)
        risk_level = prediction.get('risk_level', 'N/A')
        employee_name = prediction.get('employee_name', 'This employee')
        trend = prediction.get('risk_trend', 'stable')
        trend_icon = prediction.get('risk_trend_icon', '→')
        
        # Determine urgency
        if risk_score >= 85:
            urgency = "This requires immediate attention"
        elif risk_score >= 60:
            urgency = "This warrants proactive intervention"
        elif risk_score >= 30:
            urgency = "This suggests monitoring and support"
        else:
            urgency = "Current indicators are stable"
        
        content = f"**What the data indicates:**\n\n"
        content += f"{employee_name} has a **{risk_score:.1f}%** probabilistic risk of voluntary departure ({risk_level} risk level). "
        content += f"The risk trend is {trend} {trend_icon}.\n\n"
        
        content += f"**Why it matters now:**\n\n"
        content += f"{urgency}. "
        if trend == 'increasing':
            content += "The upward trend suggests risk factors are intensifying. "
        elif trend == 'decreasing':
            content += "Recent improvements are positive, but continued support is recommended. "
        
        content += "\n\n**What to do next:**\n\n"
        
        # Provide 2-3 concrete actions based on risk level
        if risk_score >= 85:
            content += "1. **Schedule immediate 1-on-1** with their manager to understand concerns\n"
            content += "2. **Review top risk drivers** and address root causes (career progression, compensation, engagement)\n"
            content += "3. **Offer retention support** such as development opportunities or role adjustments"
        elif risk_score >= 60:
            content += "1. **Proactive check-in** with manager to discuss career goals and satisfaction\n"
            content += "2. **Address primary risk drivers** identified in their profile\n"
            content += "3. **Monitor engagement** and provide regular feedback"
        else:
            content += "1. **Maintain regular check-ins** to ensure continued satisfaction\n"
            content += "2. **Monitor for changes** in risk indicators\n"
            content += "3. **Support career development** to prevent future risk"
        
        return {
            'type': 'text',
            'content': content,
            'data': prediction
        }
    
    def _explain_risk_drivers(self, prediction: Dict, context: Dict) -> Dict[str, Any]:
        """Explain risk drivers in plain language."""
        drivers = prediction.get('top_3_drivers', [])
        employee_name = prediction.get('employee_name', 'This employee')
        
        if not drivers:
            return {
                'type': 'text',
                'content': f"{employee_name} shows no significant risk drivers at this time. This is a positive indicator of retention likelihood."
            }
        
        content = f"**What the data indicates:**\n\n"
        content += f"The primary factors contributing to {employee_name}'s risk are:\n\n"
        
        for i, driver in enumerate(drivers[:3], 1):
            content += f"{i}. **{driver['name']}** ({driver['influence_pct']}% influence)\n"
            content += f"   {driver['description']}\n\n"
        
        content += "**Why it matters now:**\n\n"
        content += "These factors, when combined, create a pattern that suggests potential dissatisfaction. "
        content += "Addressing the top driver alone could significantly reduce overall risk.\n\n"
        
        content += "**What to do next:**\n\n"
        top_driver = drivers[0]
        if 'career' in top_driver['name'].lower() or 'progression' in top_driver['name'].lower():
            content += "1. **Discuss career path** and create a development plan\n"
            content += "2. **Identify growth opportunities** within the organization\n"
            content += "3. **Set clear milestones** for advancement"
        elif 'compensation' in top_driver['name'].lower() or 'salary' in top_driver['name'].lower():
            content += "1. **Review compensation** against market benchmarks\n"
            content += "2. **Discuss total rewards** including benefits and development\n"
            content += "3. **Explore non-monetary recognition** and growth opportunities"
        elif 'engagement' in top_driver['name'].lower() or 'manager' in top_driver['name'].lower():
            content += "1. **Increase manager 1-on-1 frequency** and quality\n"
            content += "2. **Improve communication** and feedback loops\n"
            content += "3. **Strengthen manager-employee relationship**"
        else:
            content += "1. **Address the specific concern** identified in the risk driver\n"
            content += "2. **Provide targeted support** based on the driver category\n"
            content += "3. **Monitor improvement** in this area"
        
        return {
            'type': 'text',
            'content': content,
            'data': prediction
        }
    
    def _recommend_actions(self, prediction: Dict, context: Dict) -> Dict[str, Any]:
        """Provide actionable recommendations."""
        risk_score = prediction.get('risk_score', 0)
        drivers = prediction.get('top_3_drivers', [])
        employee_name = prediction.get('employee_name', 'This employee')
        
        content = f"**Recommended actions for {employee_name}:**\n\n"
        
        if risk_score >= 85:
            content += "**Immediate Actions (This Week):**\n"
            content += "1. Manager should schedule a confidential conversation to understand concerns\n"
            content += "2. HR partner should review retention options and support resources\n"
            content += "3. Address the top risk driver with a concrete plan\n\n"
            
            content += "**Short-term Actions (This Month):**\n"
            content += "1. Implement retention interventions based on risk drivers\n"
            content += "2. Provide development opportunities or role adjustments\n"
            content += "3. Increase recognition and feedback frequency\n"
            
        elif risk_score >= 60:
            content += "**Proactive Actions:**\n"
            content += "1. Schedule a check-in conversation to discuss satisfaction and goals\n"
            content += "2. Address primary risk drivers with targeted support\n"
            content += "3. Create a development plan aligned with their career interests\n\n"
            
            content += "**Ongoing Support:**\n"
            content += "1. Monitor engagement and performance trends\n"
            content += "2. Provide regular feedback and recognition\n"
            content += "3. Ensure clear communication about opportunities"
            
        else:
            content += "**Preventive Actions:**\n"
            content += "1. Maintain regular check-ins and open communication\n"
            content += "2. Support career development and growth opportunities\n"
            content += "3. Monitor for changes in risk indicators\n\n"
            
            content += "**Best Practices:**\n"
            content += "1. Ensure manager provides regular feedback\n"
            content += "2. Recognize contributions and achievements\n"
            content += "3. Keep career conversations ongoing"
        
        if drivers:
            top_driver = drivers[0]
            content += f"\n\n**Focus Area:** Prioritize addressing **{top_driver['name']}** "
            content += f"as it has {top_driver['influence_pct']}% influence on overall risk."
        
        return {
            'type': 'text',
            'content': content,
            'data': prediction
        }
    
    def _explain_trend(self, prediction: Dict, context: Dict) -> Dict[str, Any]:
        """Explain risk trend."""
        trend = prediction.get('risk_trend', 'stable')
        trend_icon = prediction.get('risk_trend_icon', '→')
        employee_name = prediction.get('employee_name', 'This employee')
        
        content = f"**What the data indicates:**\n\n"
        
        if trend == 'increasing':
            content += f"{employee_name}'s risk trend is **increasing** {trend_icon}, "
            content += "indicating that risk factors are intensifying over time.\n\n"
            content += "**Why it matters now:**\n\n"
            content += "An upward trend suggests that without intervention, the risk of departure is likely to continue rising. "
            content += "Early action can help reverse this trajectory.\n\n"
            content += "**What to do next:**\n\n"
            content += "1. **Investigate recent changes** that may have contributed to the increase\n"
            content += "2. **Address root causes** before risk escalates further\n"
            content += "3. **Increase support and engagement** to stabilize the trend"
            
        elif trend == 'decreasing':
            content += f"{employee_name}'s risk trend is **decreasing** {trend_icon}, "
            content += "indicating that risk factors are improving.\n\n"
            content += "**Why it matters now:**\n\n"
            content += "A downward trend is positive, but it's important to maintain the momentum. "
            content += "Continued support will help ensure the trend remains positive.\n\n"
            content += "**What to do next:**\n\n"
            content += "1. **Continue current interventions** that are showing positive results\n"
            content += "2. **Maintain regular check-ins** to ensure continued improvement\n"
            content += "3. **Reinforce positive changes** and recognize progress"
            
        else:
            content += f"{employee_name}'s risk trend is **stable** {trend_icon}, "
            content += "indicating that risk factors have remained relatively consistent.\n\n"
            content += "**Why it matters now:**\n\n"
            content += "Stability can be positive for low-risk employees, but for higher-risk individuals, "
            content += "it suggests that underlying concerns may persist.\n\n"
            content += "**What to do next:**\n\n"
            content += "1. **Monitor for changes** in risk indicators\n"
            content += "2. **Address any persistent risk drivers**\n"
            content += "3. **Proactively support** to prevent future increases"
        
        return {
            'type': 'text',
            'content': content,
            'data': prediction
        }
    
    def _suggest_conversation_approach(self, prediction: Dict, context: Dict) -> Dict[str, Any]:
        """Suggest how to approach a conversation with the employee."""
        risk_score = prediction.get('risk_score', 0)
        drivers = prediction.get('top_3_drivers', [])
        employee_name = prediction.get('employee_name', 'This employee')
        
        content = f"**Suggested approach for {employee_name}:**\n\n"
        
        content += "**Conversation Framework:**\n\n"
        content += "1. **Start with open-ended questions** about their experience and satisfaction\n"
        content += "2. **Listen actively** to understand their perspective and concerns\n"
        content += "3. **Acknowledge their contributions** and value to the organization\n"
        content += "4. **Discuss opportunities** for growth, development, or role adjustments\n"
        content += "5. **Create an action plan** together with clear next steps\n\n"
        
        if drivers:
            top_driver = drivers[0]
            content += "**Focus Areas Based on Risk Drivers:**\n\n"
            
            if 'career' in top_driver['name'].lower():
                content += "• Ask about their career goals and aspirations\n"
                content += "• Discuss development opportunities and growth paths\n"
                content += "• Explore ways to provide more challenging assignments\n"
            elif 'compensation' in top_driver['name'].lower():
                content += "• Understand their perspective on total compensation\n"
                content += "• Discuss non-monetary rewards and recognition\n"
                content += "• Explore development opportunities as value-add\n"
            elif 'engagement' in top_driver['name'].lower() or 'manager' in top_driver['name'].lower():
                content += "• Ask about their relationship with their manager\n"
                content += "• Discuss communication preferences and feedback needs\n"
                content += "• Explore ways to improve collaboration and support\n"
            else:
                content += f"• Address concerns related to {top_driver['name']}\n"
                content += "• Understand their specific needs and challenges\n"
                content += "• Work together to find solutions\n"
        
        content += "\n**Important:** Frame this as a supportive conversation, not an interrogation. "
        content += "The goal is to understand their needs and provide support, not to pressure them to stay."
        
        return {
            'type': 'text',
            'content': content,
            'data': prediction
        }
    
    def _handle_risk_overview(self, query: str, context: Dict, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Handle risk overview queries in dashboard context."""
        summary = get_risk_summary(self.df)
        query_lower = query.lower()
        is_followup = conversation_history and len(conversation_history) > 2
        
        if 'critical' in query_lower:
            count = summary['critical_count']
            total = summary['total_employees']
            pct = (count / total) * 100
            
            if is_followup:
                content = f"There are **{count} employees** ({pct:.1f}%) at critical risk (85%+). "
                content += "These need immediate attention. Should I help you identify who they are?"
            else:
            content = "**What the data indicates:**\n\n"
            content += f"There are **{count} employees** ({pct:.1f}%) at critical risk (85%+ risk score).\n\n"
            content += "**Why it matters now:**\n\n"
            content += "Critical risk employees require immediate attention. "
            content += "Without intervention, there's a high probability of voluntary departure.\n\n"
            content += "**What to do next:**\n\n"
            content += "1. **Prioritize these employees** for immediate retention conversations\n"
            content += "2. **Review their risk drivers** to identify root causes\n"
            content += "3. **Develop targeted retention plans** for each individual"
            
            return {
                'type': 'text',
                'content': content,
                'data': summary
            }
        
        elif 'high' in query_lower:
            count = summary['high_risk_count']
            total = summary['total_employees']
            pct = (count / total) * 100
            
            if is_followup:
                content = f"**{count} employees** ({pct:.1f}%) are at high or critical risk. "
                content += "Would you like to see the breakdown by department?"
            else:
            content = "**What the data indicates:**\n\n"
            content += f"There are **{count} employees** ({pct:.1f}%) at high or critical risk (60%+ risk score).\n\n"
            content += "**Why it matters now:**\n\n"
            content += "This represents a significant portion of the workforce requiring proactive intervention. "
            content += "Early action can prevent further escalation.\n\n"
            content += "**What to do next:**\n\n"
            content += "1. **Review the high-risk employee list** to identify patterns\n"
            content += "2. **Prioritize by department** or role criticality\n"
            content += "3. **Develop department-level retention strategies**"
            
            return {
                'type': 'text',
                'content': content,
                'data': summary
            }
        
        # General risk overview
        if is_followup:
            content = f"Overall risk score is **{summary['avg_risk_score']:.1f}%**. "
            content += f"**{summary['high_risk_count']} employees** ({summary['high_risk_count']/summary['total_employees']*100:.1f}%) are at high/critical risk. "
            content += "Want more details?"
        else:
        content = "**What the data indicates:**\n\n"
        content += f"Organization-wide average risk score is **{summary['avg_risk_score']:.1f}%**. "
        content += f"{summary['high_risk_count']} employees ({summary['high_risk_count']/summary['total_employees']*100:.1f}%) "
        content += f"are at high or critical risk.\n\n"
        
        content += "**Why it matters now:**\n\n"
        content += "Understanding overall risk helps prioritize retention efforts and identify systemic issues.\n\n"
        
        content += "**What to do next:**\n\n"
        content += "1. **Focus on high-risk employees** first for immediate impact\n"
        content += "2. **Analyze department patterns** to identify organizational factors\n"
        content += "3. **Develop targeted retention programs** based on common risk drivers"
        
        return {
            'type': 'text',
            'content': content,
            'data': summary
        }
    
    def _handle_department_analysis(self, query: str, context: Dict, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Handle department analysis queries."""
        query_lower = query.lower()
        dept_summary = get_department_summary(self.df)
        
        # Extract department name
        dept_name = None
        for dept in dept_summary.keys():
            if dept.lower() in query_lower:
                dept_name = dept
                break
        
        # Fuzzy match if no exact match
        if not dept_name:
            dept_match = process.extractOne(
                query_lower,
                self.departments,
                scorer=fuzz.partial_ratio,
                score_cutoff=60
            )
            if dept_match:
                dept_name = dept_match[0]
        
        is_followup = conversation_history and len(conversation_history) > 2
        
        if dept_name and dept_name in dept_summary:
            stats = dept_summary[dept_name]
            high_risk_count = stats['distribution'].get('High', 0) + stats['distribution'].get('Critical', 0)
            
            if is_followup:
                content = f"The **{dept_name}** department has an average risk of **{stats['avg_score']:.1f}%**. "
                content += f"{high_risk_count} employees ({stats['high_risk_pct']:.1f}%) are at high/critical risk. "
                if stats['avg_score'] > 40:
                    content += "This is elevated - want to see what's driving it?"
                else:
                    content += "Risk levels look manageable. Need more details?"
            else:
            content = f"**What the data indicates:**\n\n"
            content += f"The **{dept_name}** department has an average risk score of **{stats['avg_score']:.1f}%**. "
            content += f"{high_risk_count} employees ({stats['high_risk_pct']:.1f}%) are at high or critical risk.\n\n"
            
            content += "**Why it matters now:**\n\n"
            if stats['avg_score'] > 40:
                content += "This department shows elevated risk levels, suggesting potential systemic issues "
                content += "that may affect multiple employees.\n\n"
            else:
                content += "While risk levels are manageable, proactive monitoring can help prevent escalation.\n\n"
            
            content += "**What to do next:**\n\n"
            content += "1. **Review department-specific risk drivers** to identify common factors\n"
            content += "2. **Engage department leadership** in retention planning\n"
            content += "3. **Develop targeted interventions** for high-risk individuals"
            
            return {
                'type': 'text',
                'content': content,
                'data': stats
            }
        
        # Which department has highest risk
        if 'highest' in query_lower or 'most' in query_lower:
            highest_dept = max(dept_summary.items(), key=lambda x: x[1]['avg_score'])
            stats = highest_dept[1]
            
            if is_followup:
                content = f"**{highest_dept[0]}** has the highest risk at **{stats['avg_score']:.1f}%**. "
                content += "Should I analyze what's causing this?"
            else:
            content = "**What the data indicates:**\n\n"
            content += f"**{highest_dept[0]}** has the highest average risk score at **{stats['avg_score']:.1f}%**.\n\n"
            
            content += "**Why it matters now:**\n\n"
            content += "This department requires focused attention to understand and address underlying risk factors.\n\n"
            
            content += "**What to do next:**\n\n"
            content += "1. **Investigate department-specific factors** contributing to risk\n"
            content += "2. **Engage with department leadership** to develop retention strategies\n"
            content += "3. **Prioritize high-risk employees** in this department for intervention"
            
            return {
                'type': 'text',
                'content': content,
                'data': stats
            }
        
        return {
            'type': 'text',
            'content': "I can analyze departments. Try asking:\n\n"
                      "• 'Which department has the highest risk?'\n"
                      "• 'Engineering department analysis'\n"
                      "• 'Show me Sales department statistics'",
            'suggestions': ["Highest risk department", "Department comparison", "Engineering analysis"]
        }
    
    def _handle_statistics(self, query: str, context: Dict, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Handle statistics queries."""
        summary = get_risk_summary(self.df)
        
        # Check if this is a follow-up question
        is_followup = conversation_history and len(conversation_history) > 2
        
        if is_followup:
            # Shorter, more conversational response
            content = f"Here are the key numbers:\n\n"
            content += f"• **{summary['total_employees']}** total employees\n"
            content += f"• **{summary['high_risk_count']}** at high/critical risk ({summary['high_risk_count']/summary['total_employees']*100:.1f}%)\n"
            content += f"• **{summary['critical_count']}** at critical risk ({summary['critical_count']/summary['total_employees']*100:.1f}%)\n"
            content += f"• Average risk score: **{summary['avg_risk_score']:.1f}%**\n\n"
            content += "Would you like more details about any specific area?"
        else:
            # Full response for first-time queries
        content = "**What the data indicates:**\n\n"
        content += f"• Total Employees: **{summary['total_employees']}**\n"
        content += f"• High Risk Employees: **{summary['high_risk_count']}** ({summary['high_risk_count']/summary['total_employees']*100:.1f}%)\n"
        content += f"• Critical Risk: **{summary['critical_count']}** ({summary['critical_count']/summary['total_employees']*100:.1f}%)\n"
        content += f"• Average Risk Score: **{summary['avg_risk_score']:.1f}%**\n\n"
        
        content += "**Risk Distribution:**\n"
        for level, count in summary['risk_distribution'].items():
            pct = (count / summary['total_employees']) * 100
            content += f"• {level}: {count} ({pct:.1f}%)\n"
        
        content += "\n**Why it matters now:**\n\n"
        content += "These metrics provide an organizational overview of retention risk. "
        content += "Focusing on high-risk employees can help prevent voluntary departures.\n\n"
        
        content += "**What to do next:**\n\n"
        content += "1. **Prioritize critical and high-risk employees** for immediate attention\n"
        content += "2. **Review department patterns** to identify systemic issues\n"
        content += "3. **Develop retention strategies** based on common risk drivers"
        
        return {
            'type': 'text',
            'content': content,
            'data': summary
        }
    
    def _handle_employee_search(self, query: str, context: Dict, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Handle employee search queries."""
        # Extract search term
        search_term = query
        for word in ['find', 'search', 'show', 'who is', 'employee']:
            search_term = re.sub(f'^{word}\\s+', '', search_term, flags=re.IGNORECASE)
        
        search_term = search_term.strip()
        
        if not search_term:
            return {
                'type': 'text',
                'content': "Please specify which employee you'd like to find. Try:\n\n"
                          "• 'Find John Smith'\n"
                          "• 'Search EMP001'\n"
                          "• 'Who is [Employee Name]'"
            }
        
        # Use fuzzy search
        results = self.search_employees(search_term, limit=5, min_score=50)
        
        if len(results) == 1:
            result = results[0]
            content = f"**Found:** {result['name']} ({result['id']})\n\n"
            content += f"**Department:** {result['department']}\n"
            content += f"**Risk Score:** {result['risk_score']:.1f}% ({result['risk_level']})\n\n"
            content += "**What to do next:**\n\n"
            content += f"1. **View their profile** to see detailed risk analysis\n"
            content += f"2. **Review their risk drivers** to understand contributing factors\n"
            content += f"3. **Develop a retention plan** if they're at high risk"
            
            return {
                'type': 'employee_detail',
                'content': content,
                'employee_id': result['id'],
                'data': result
            }
        
        elif len(results) > 1:
            content = f"Found {len(results)} employees matching '{search_term}':\n\n"
            for r in results[:5]:
                content += f"• **{r['name']}** ({r['id']}) - {r['department']} - {r['risk_score']:.1f}%\n"
            
            content += "\n**What to do next:**\n\n"
            content += "1. **Refine your search** with a more specific name or ID\n"
            content += "2. **Select an employee** from the list to view their profile\n"
            content += "3. **Use filters** to narrow down results"
            
            return {
                'type': 'employee_list',
                'content': content,
                'data': results
            }
        
        return {
            'type': 'text',
            'content': f"Could not find employees matching '{search_term}'. Try:\n\n"
                      "• Using a different spelling or partial name\n"
                      "• Searching by employee ID (e.g., EMP001)\n"
                      "• Using the Search page for advanced filtering"
        }
    
    def _handle_comparison(self, query: str, context: Dict, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Handle comparison queries."""
        dept_summary = get_department_summary(self.df)
        query_lower = query.lower()
        
        # Extract department names
        depts = []
        for dept in dept_summary.keys():
            if dept.lower() in query_lower:
                depts.append(dept)
        
        if len(depts) >= 2:
            content = "**Department Comparison:**\n\n"
            for dept in depts[:3]:
                stats = dept_summary[dept]
                content += f"**{dept}:**\n"
                content += f"• Avg Risk: {stats['avg_score']:.1f}%\n"
                content += f"• High Risk: {stats['high_risk_pct']:.1f}%\n"
                content += f"• Total: {stats['total']} employees\n\n"
            
            content += "**What to do next:**\n\n"
            content += "1. **Identify differences** in risk drivers between departments\n"
            content += "2. **Learn from lower-risk departments** what practices work well\n"
            content += "3. **Apply best practices** to higher-risk departments"
            
            return {
                'type': 'text',
                'content': content
            }
        
        return {
            'type': 'text',
            'content': "I can compare departments. Try:\n\n"
                      "• 'Compare Engineering and Sales'\n"
                      "• 'Engineering vs Finance'\n"
                      "• 'Department comparison'"
        }
    
    def _handle_prioritization(self, query: str, context: Dict, conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Handle prioritization queries."""
        summary = get_risk_summary(self.df)
        high_risk = get_high_risk_employees(self.df, min_level='high')
        
        content = "**What the data indicates:**\n\n"
        content += f"There are **{summary['critical_count']} critical-risk** and **{summary['high_risk_count'] - summary['critical_count']} high-risk** employees requiring attention.\n\n"
        
        content += "**Why it matters now:**\n\n"
        content += "Prioritizing interventions can maximize retention impact with limited resources.\n\n"
        
        content += "**What to do next:**\n\n"
        content += "1. **Start with critical-risk employees** (85%+ risk score) - these require immediate action\n"
        content += "2. **Focus on high-value employees** - consider role criticality and performance\n"
        content += "3. **Address department clusters** - if multiple high-risk employees are in one department, investigate systemic issues\n"
        content += "4. **Review common risk drivers** - address root causes that affect multiple employees"
        
        if high_risk:
            content += "\n\n**Top Priority Employees:**\n"
            for emp in high_risk[:5]:
                content += f"• {emp['employee_name']} ({emp['employee_id']}) - {emp['risk_score']:.1f}%\n"
        
        return {
            'type': 'text',
            'content': content,
            'data': {'high_risk': high_risk[:10]}
        }
    
    def _get_contextual_help(self, context: Dict) -> Dict[str, Any]:
        """Provide contextual help based on current view."""
        view_type = context.get('view_type', 'dashboard')
        
        if view_type == 'employee_profile':
            return {
                'type': 'text',
                'content': "**I can help you understand this employee's risk profile.**\n\n"
                          "Try asking:\n"
                          "• 'What's their risk score?'\n"
                          "• 'Why are they at risk?'\n"
                          "• 'What should I do?'\n"
                          "• 'How should I approach a conversation?'\n"
                          "• 'What's the trend?'",
                'suggestions': [
                    "What's their risk score?",
                    "Why are they at risk?",
                    "What should I do?"
                ]
            }
        elif view_type == 'chatbot':
            return {
                'type': 'text',
                'content': "**I'm your AI Assistant for HR Risk Analysis.**\n\n"
                          "I can help you with:\n\n"
                          "**Data Analysis:**\n"
                          "• 'Show me risk statistics'\n"
                          "• 'Which department has the highest risk?'\n"
                          "• 'Analyze patterns in the data'\n\n"
                          "**Visualizations:**\n"
                          "• 'Create a chart comparing departments'\n"
                          "• 'Show me a risk distribution chart'\n"
                          "• 'Visualize high risk employees'\n\n"
                          "• 'What recommendations do you have?'\n"
                          "• 'Find [Employee Name]'\n"
                          "• 'Compare departments'",
                'suggestions': [
                    "Show risk statistics",
                    "Create a department chart",
                    "What recommendations do you have?"
                ]
            }
        else:
            return {
                'type': 'text',
                'content': "**I can help you analyze organizational risk.**\n\n"
                          "Try asking:\n"
                          "• 'How many employees are at high risk?'\n"
                          "• 'Which department has the highest risk?'\n"
                          "• 'Show me risk statistics'\n"
                          "• 'What should we prioritize?'\n"
                          "• 'Find [Employee Name]'",
                'suggestions': [
                    "How many at high risk?",
                    "Highest risk department",
                    "Risk statistics"
                ]
            }
    
    def search_employees(self, search_term: str, limit: int = 10, min_score: int = 50) -> List[Dict[str, Any]]:
        """
        Enhanced search for employees by name or ID with fuzzy matching and ranking.
        """
        if not search_term or len(search_term) < 2:
            return []
        
        results = []
        search_term_lower = search_term.lower()
        
        # Search by name with fuzzy matching
        name_matches = []
        for name in self.employee_names:
            ratio_score = fuzz.ratio(search_term_lower, name.lower())
            partial_score = fuzz.partial_ratio(search_term_lower, name.lower())
            token_score = fuzz.token_sort_ratio(search_term_lower, name.lower())
            
            best_score = max(ratio_score, partial_score, token_score)
            
            if best_score >= min_score:
                name_matches.append((name, best_score))
        
        # Search by ID
        id_matches = []
        for emp_id in self.employee_ids:
            if search_term_lower in emp_id.lower():
                score = 100 if search_term_lower == emp_id.lower() else 80
                id_matches.append((emp_id, score))
        
        # Combine and sort by score
        all_matches = name_matches + id_matches
        all_matches.sort(key=lambda x: x[1], reverse=True)
        
        # Get unique employees
        seen = set()
        for match_item, score in all_matches[:limit * 2]:
            if match_item in seen:
                continue
            seen.add(match_item)
            
            if match_item.startswith('EMP'):
                employee_row = self.df[self.df['Employee_ID'] == match_item]
            else:
                employee_row = self.df[self.df['Name'] == match_item]
            
            if not employee_row.empty:
                emp = employee_row.iloc[0]
                prediction = predict_attrition_risk(emp)
                results.append({
                    'name': prediction['employee_name'],
                    'id': prediction['employee_id'],
                    'department': prediction['department'],
                    'risk_score': prediction['risk_score'],
                    'risk_level': prediction['risk_level'],
                    'match_score': score
                })
                
                if len(results) >= limit:
                    break
        
        return results
    
    def get_autocomplete_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get autocomplete suggestions for a partial query."""
        if not partial_query or len(partial_query) < 2:
            return []
        
        suggestions = []
        partial_lower = partial_query.lower()
        
        for name in self.employee_names:
            if partial_lower in name.lower():
                suggestions.append(f"Find {name}")
                if len(suggestions) >= limit:
                    break
        
        for dept in self.departments:
            if partial_lower in dept.lower():
                suggestions.append(f"Show {dept} department")
                if len(suggestions) >= limit * 2:
                    break
        
        return suggestions[:limit]
    
    # =============================================================================
    # ENHANCED CAPABILITIES: Chart Generation, Pattern Analysis, Recommendations
    # =============================================================================
    
    def _handle_chart_request(self, query: str, context: Dict[str, Any], conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Handle chart generation requests."""
        query_lower = query.lower()
        
        # Department comparison chart
        if any(word in query_lower for word in ['department', 'dept', 'compare']):
            return self._generate_department_chart(context)
        
        # Risk distribution chart
        if any(word in query_lower for word in ['risk', 'distribution', 'overview']):
            return self._generate_risk_distribution_chart(context)
        
        # High risk employees chart
        if any(word in query_lower for word in ['high risk', 'critical']):
            return self._generate_high_risk_chart(context)
        
        # Default: Department comparison
        return self._generate_department_chart(context)
    
    def _generate_department_chart(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate department comparison chart."""
        dept_summary = get_department_summary(self.df)
        dept_data = []
        for dept, stats in dept_summary.items():
            dept_data.append({
                'Department': dept,
                'Avg Risk Score': stats['avg_score'],
                'High Risk %': stats['high_risk_pct']
            })
        dept_df = pd.DataFrame(dept_data)
        
        fig = go.Figure(data=[
            go.Bar(
                x=dept_df['Department'],
                y=dept_df['Avg Risk Score'],
                marker=dict(
                    color=dept_df['Avg Risk Score'],
                    colorscale='RdYlGn_r',
                    showscale=True,
                    colorbar=dict(title="Risk Score")
                ),
                text=[f"{x:.1f}%" for x in dept_df['Avg Risk Score']],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Avg Risk: %{y:.1f}%<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="Department Risk Comparison",
            xaxis_title="Department",
            yaxis_title="Average Risk Score (%)",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return {
            'type': 'chart',
            'content': "Here's a comparison of risk scores across departments:",
            'chart_data': fig
        }
    
    def _generate_risk_distribution_chart(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate risk distribution chart."""
        summary = get_risk_summary(self.df)
        risk_dist = summary['risk_distribution']
        
        fig = go.Figure(data=[go.Pie(
            labels=list(risk_dist.keys()),
            values=list(risk_dist.values()),
            hole=0.4,
            marker=dict(
                colors=['#28A745', '#FFC107', '#FF9800', '#DC3545'],
                line=dict(color='#FFFFFF', width=2)
            ),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Employees: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Risk Distribution Across Organization",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return {
            'type': 'chart',
            'content': "Here's the risk distribution across all employees:",
            'chart_data': fig
        }
    
    def _generate_high_risk_chart(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high risk employees chart."""
        high_risk = get_high_risk_employees(self.df, min_level='high')
        
        if not high_risk:
            return {
                'type': 'text',
                'content': "No high-risk employees found in the dataset."
            }
        
        # Group by department
        dept_counts = {}
        for emp in high_risk[:20]:  # Limit to top 20
            dept = emp.get('department', 'Unknown')
            dept_counts[dept] = dept_counts.get(dept, 0) + 1
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(dept_counts.keys()),
                y=list(dept_counts.values()),
                marker=dict(color='#DC3545'),
                hovertemplate='<b>%{x}</b><br>High Risk Employees: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title="High Risk Employees by Department",
            xaxis_title="Department",
            yaxis_title="Number of High Risk Employees",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return {
            'type': 'chart',
            'content': f"Here's the distribution of {len(high_risk)} high-risk employees across departments:",
            'chart_data': fig
        }
    
    def _handle_pattern_analysis(self, query: str, context: Dict[str, Any], conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Handle pattern analysis queries."""
        query_lower = query.lower()
        is_followup = conversation_history and len(conversation_history) > 2
        
        # Analyze patterns across employees
        patterns = self._analyze_patterns()
        
        if is_followup:
            # Shorter, more conversational response
            content = "Here's what I found:\n\n"
            if patterns.get('driver_patterns'):
                top_driver = list(patterns['driver_patterns'].keys())[0]
                content += f"The most common issue is **{top_driver}**, affecting {patterns['driver_patterns'][top_driver]} employees.\n\n"
            if patterns.get('insights'):
                content += patterns['insights'][0] + "\n\n"
            content += "Want me to dive deeper into any specific pattern?"
        else:
            content = "**Pattern Analysis Results:**\n\n"
            
            # Department patterns
            if patterns.get('department_patterns'):
                content += "**Department-Level Patterns:**\n\n"
                for dept, info in patterns['department_patterns'].items():
                    content += f"• **{dept}**: {info['description']}\n"
                content += "\n"
            
            # Risk driver patterns
            if patterns.get('driver_patterns'):
                content += "**Common Risk Drivers:**\n\n"
                for driver, count in patterns['driver_patterns'].items():
                    content += f"• **{driver}**: Affects {count} employees\n"
                content += "\n"
            
            # Correlation insights
            if patterns.get('insights'):
                content += "**Key Insights:**\n\n"
                for insight in patterns['insights']:
                    content += f"• {insight}\n"
        
        return {
            'type': 'text',
            'content': content,
            'data': patterns
        }
    
    def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze patterns across the dataset."""
        patterns = {
            'department_patterns': {},
            'driver_patterns': {},
            'insights': []
        }
        
        # Analyze department patterns
        dept_summary = get_department_summary(self.df)
        high_risk_depts = []
        low_risk_depts = []
        
        for dept, stats in dept_summary.items():
            if stats['avg_score'] > 50:
                high_risk_depts.append(dept)
                patterns['department_patterns'][dept] = {
                    'description': f"High average risk ({stats['avg_score']:.1f}%) with {stats['high_risk_pct']:.1f}% high-risk employees"
                }
            elif stats['avg_score'] < 25:
                low_risk_depts.append(dept)
                patterns['department_patterns'][dept] = {
                    'description': f"Low average risk ({stats['avg_score']:.1f}%) - good retention indicators"
                }
        
        # Analyze common risk drivers
        high_risk_employees = get_high_risk_employees(self.df, min_level='moderate')
        driver_counts = {}
        
        for emp in high_risk_employees[:50]:  # Sample first 50
            prediction = get_employee_by_id(self.df, emp.get('employee_id', ''))
            if prediction and prediction.get('top_3_drivers'):
                top_driver = prediction['top_3_drivers'][0]
                driver_name = top_driver.get('name', 'Unknown')
                driver_counts[driver_name] = driver_counts.get(driver_name, 0) + 1
        
        patterns['driver_patterns'] = dict(sorted(driver_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Generate insights
        summary = get_risk_summary(self.df)
        total = summary['total_employees']
        high_risk_pct = (summary['high_risk_count'] / total) * 100
        
        patterns['insights'].append(
            f"Overall {high_risk_pct:.1f}% of employees are at high or critical risk, indicating potential systemic issues"
        )
        
        if high_risk_depts:
            patterns['insights'].append(
                f"Departments with elevated risk: {', '.join(high_risk_depts[:3])}"
            )
        
        if patterns['driver_patterns']:
            top_driver = list(patterns['driver_patterns'].keys())[0]
            patterns['insights'].append(
                f"The most common risk driver is '{top_driver}', affecting {patterns['driver_patterns'][top_driver]} employees"
            )
        
        return patterns
    
    def _handle_recommendations(self, query: str, context: Dict[str, Any], conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Handle recommendation requests."""
        recommendations = self._propose_recommendations()
        is_followup = conversation_history and len(conversation_history) > 2
        
        if is_followup:
            # Shorter, more conversational response
            content = "Here are my top recommendations:\n\n"
            if recommendations.get('immediate'):
                content += "**This Week:**\n"
                for rec in recommendations['immediate'][:2]:  # Just top 2
                    content += f"• {rec}\n"
                content += "\n"
            if recommendations.get('short_term'):
                content += "**This Month:**\n"
                for rec in recommendations['short_term'][:2]:  # Just top 2
                    content += f"• {rec}\n"
            content += "\nWant more detailed action plans?"
        else:
            content = "**Strategic Recommendations:**\n\n"
            
            # Immediate actions
            if recommendations.get('immediate'):
                content += "**Immediate Actions (This Week):**\n\n"
                for i, rec in enumerate(recommendations['immediate'], 1):
                    content += f"{i}. {rec}\n"
                content += "\n"
            
            # Short-term actions
            if recommendations.get('short_term'):
                content += "**Short-term Actions (This Month):**\n\n"
                for i, rec in enumerate(recommendations['short_term'], 1):
                    content += f"{i}. {rec}\n"
                content += "\n"
            
            # Long-term strategies
            if recommendations.get('long_term'):
                content += "**Long-term Strategies:**\n\n"
                for i, rec in enumerate(recommendations['long_term'], 1):
                    content += f"{i}. {rec}\n"
        
        return {
            'type': 'text',
            'content': content,
            'data': recommendations
        }
    
    def _propose_recommendations(self) -> Dict[str, List[str]]:
        """Propose actionable HR recommendations based on data analysis."""
        recommendations = {
            'immediate': [],
            'short_term': [],
            'long_term': []
        }
        
        summary = get_risk_summary(self.df)
        dept_summary = get_department_summary(self.df)
        
        # Immediate actions based on critical risk
        if summary['critical_count'] > 0:
            recommendations['immediate'].append(
                f"Schedule retention conversations with {summary['critical_count']} critical-risk employees (85%+ risk score)"
            )
            recommendations['immediate'].append(
                "Review and address top risk drivers for critical-risk employees"
            )
        
        # Identify highest risk department
        highest_dept = max(dept_summary.items(), key=lambda x: x[1]['avg_score'])
        if highest_dept[1]['avg_score'] > 40:
            recommendations['immediate'].append(
                f"Engage with {highest_dept[0]} department leadership - highest risk department ({highest_dept[1]['avg_score']:.1f}% avg)"
            )
        
        # Short-term actions
        if summary['high_risk_count'] > 0:
            recommendations['short_term'].append(
                f"Develop retention plans for {summary['high_risk_count']} high-risk employees"
            )
        
        # Analyze common risk drivers
        patterns = self._analyze_patterns()
        if patterns.get('driver_patterns'):
            top_driver = list(patterns['driver_patterns'].keys())[0]
            recommendations['short_term'].append(
                f"Address '{top_driver}' - the most common risk driver affecting {patterns['driver_patterns'][top_driver]} employees"
            )
        
        recommendations['short_term'].append(
            "Implement department-specific retention strategies based on risk patterns"
        )
        
        # Long-term strategies
        recommendations['long_term'].append(
            "Establish regular risk monitoring and early intervention protocols"
        )
        recommendations['long_term'].append(
            "Develop proactive engagement programs to prevent risk escalation"
        )
        recommendations['long_term'].append(
            "Create data-driven retention policies based on identified patterns"
        )
        
        return recommendations
