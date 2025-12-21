"""
Context-Aware AI Assistant for HR Risk Analysis Dashboard
==========================================================
Intelligent assistant that adapts responses based on current UI context.
"""

import pandas as pd
import re
from typing import Dict, List, Any, Optional, Tuple
from rapidfuzz import fuzz, process
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
        self._build_search_index()
        
    def _build_search_index(self):
        """Build search index for faster lookups."""
        self.employee_names = self.df['Name'].tolist()
        self.employee_ids = self.df['Employee_ID'].tolist()
        self.departments = sorted(self.df['Department'].unique().tolist())
        
    def process_query(self, query: str, ui_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a natural language query with UI context awareness.
        
        Args:
            query: User's natural language query
            ui_context: Current UI context including view_type, selected employee, etc.
            
        Returns:
            Dictionary with response type, content, and optional data
        """
        if not query or not query.strip():
            return self._get_contextual_help(ui_context)
        
        query_lower = query.lower().strip()
        ui_context = ui_context or {}
        view_type = ui_context.get('view_type', 'dashboard')
        
        # Context-aware processing
        if view_type == 'employee_profile':
            return self._handle_employee_context_query(query, ui_context)
        else:
            return self._handle_dashboard_context_query(query, ui_context)
    
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
    
    def _handle_dashboard_context_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle queries in dashboard context."""
        query_lower = query.lower()
        
        # Risk-related queries
        if any(word in query_lower for word in ['high risk', 'critical', 'at risk', 'risk']):
            return self._handle_risk_overview(query, context)
        
        # Department queries
        if any(word in query_lower for word in ['department', 'dept', 'team', 'division']):
            return self._handle_department_analysis(query, context)
        
        # Statistics queries
        if any(word in query_lower for word in ['how many', 'total', 'statistics', 'summary', 'overview']):
            return self._handle_statistics(query, context)
        
        # Employee search
        if any(word in query_lower for word in ['find', 'search', 'employee', 'who is']):
            return self._handle_employee_search(query, context)
        
        # Comparison queries
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']):
            return self._handle_comparison(query, context)
        
        # Priority/focus queries
        if any(word in query_lower for word in ['priority', 'focus', 'urgent', 'important', 'next']):
            return self._handle_prioritization(query, context)
        
        # Default: Provide dashboard overview
        return self._handle_statistics(query, context)
    
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
    
    def _handle_risk_overview(self, query: str, context: Dict) -> Dict[str, Any]:
        """Handle risk overview queries in dashboard context."""
        summary = get_risk_summary(self.df)
        query_lower = query.lower()
        
        if 'critical' in query_lower:
            count = summary['critical_count']
            total = summary['total_employees']
            pct = (count / total) * 100
            
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
    
    def _handle_department_analysis(self, query: str, context: Dict) -> Dict[str, Any]:
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
        
        if dept_name and dept_name in dept_summary:
            stats = dept_summary[dept_name]
            high_risk_count = stats['distribution'].get('High', 0) + stats['distribution'].get('Critical', 0)
            
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
    
    def _handle_statistics(self, query: str, context: Dict) -> Dict[str, Any]:
        """Handle statistics queries."""
        summary = get_risk_summary(self.df)
        
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
    
    def _handle_employee_search(self, query: str, context: Dict) -> Dict[str, Any]:
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
    
    def _handle_comparison(self, query: str, context: Dict) -> Dict[str, Any]:
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
    
    def _handle_prioritization(self, query: str, context: Dict) -> Dict[str, Any]:
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
