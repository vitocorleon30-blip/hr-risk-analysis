"""
LLM-Powered HR Analytics Assistant
===================================
Uses Groq API (Llama 3.3 70B) with employee data context injection
for intelligent HR risk analysis and recommendations.
"""

import pandas as pd
import json
import os
import hashlib
from typing import Dict, List, Any, Optional, Iterator
from openai import OpenAI
from predict import (
    get_risk_summary,
    get_department_summary,
    get_high_risk_employees,
    predict_attrition_risk
)
import re


class LLMHRChatbot:
    """
    LLM-powered HR Analytics Assistant using Groq API.
    Provides intelligent responses with access to employee risk data.
    """
    
    def __init__(self, df: pd.DataFrame, api_key: Optional[str] = None):
        """
        Initialize LLM chatbot.
        
        Args:
            df: Employee DataFrame
            api_key: Groq API key (if None, will try to load from config)
        """
        self.df = df
        self.api_key = api_key or self._load_api_key()
        self.client = None
        
        # Caching for data context
        self._cached_context = None
        self._context_hash = None
        
        if self.api_key:
            try:
                self.client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=self.api_key
                )
            except Exception as e:
                print(f"Warning: Failed to initialize Groq client: {e}")
                self.client = None
        else:
            print("Warning: No Groq API key found. Chatbot will use fallback responses.")
    
    def _load_api_key(self) -> Optional[str]:
        """Load API key from config.env file."""
        try:
            config_path = os.path.join(
                os.path.dirname(__file__),
                "config.env"
            )
            
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    for line in f:
                        if line.startswith('GROQ_API_KEY='):
                            return line.split('=', 1)[1].strip()
        except Exception as e:
            print(f"Warning: Could not load API key from config: {e}")
        
        return None
    
    def _compute_df_hash(self) -> str:
        """Compute hash of DataFrame to detect changes."""
        try:
            # Create a hash from DataFrame shape and a sample of data
            df_str = f"{self.df.shape}_{self.df.head(10).to_string()}"
            return hashlib.md5(df_str.encode()).hexdigest()
        except Exception:
            return "unknown"
    
    def _build_data_context(self, force_rebuild: bool = False) -> str:
        """
        Build comprehensive data context summary for LLM.
        Uses caching to avoid rebuilding on every query.
        
        Args:
            force_rebuild: Force rebuild even if cached
            
        Returns:
            Formatted string with key statistics and insights
        """
        # Check if we need to rebuild
        current_hash = self._compute_df_hash()
        
        if not force_rebuild and self._cached_context and self._context_hash == current_hash:
            return self._cached_context
        
        try:
            # Get risk summary
            risk_summary = get_risk_summary(self.df)
            
            # Get department summary
            dept_summary = get_department_summary(self.df)
            
            # Get high-risk employees (top 10)
            high_risk = get_high_risk_employees(self.df, min_level='High')[:10]
            
            # Format context
            risk_dist = risk_summary.get('risk_distribution', {})
            context_parts = [
                "## Current Employee Risk Data Summary\n\n",
                f"**Total Employees:** {risk_summary['total_employees']}\n",
                f"**Average Risk Score:** {risk_summary['avg_risk_score']:.1f}%\n",
                f"**Risk Distribution:**\n",
                f"- Low Risk (0-30%): {risk_dist.get('Low', 0)} employees\n",
                f"- Moderate Risk (30-60%): {risk_dist.get('Moderate', 0)} employees\n",
                f"- High Risk (60-85%): {risk_dist.get('High', 0)} employees\n",
                f"- Critical Risk (85-100%): {risk_dist.get('Critical', 0)} employees\n\n",
                "## Department Analysis\n\n"
            ]
            
            # Add department data
            for dept_name, dept_data in sorted(dept_summary.items(), 
                                             key=lambda x: x[1]['avg_score'], 
                                             reverse=True)[:5]:
                context_parts.append(
                    f"- **{dept_name}**: "
                    f"Avg Risk {dept_data['avg_score']:.1f}%, "
                    f"{dept_data['high_risk_pct']:.1f}% high-risk employees, "
                    f"{dept_data['total']} total employees\n"
                )
            
            # Add high-risk employees
            if high_risk:
                context_parts.append("\n## Top High-Risk Employees\n\n")
                for i, emp in enumerate(high_risk[:5], 1):
                    emp_name = emp.get('employee_name', 'Unknown')
                    risk_score = emp.get('risk_score', 0)
                    dept = emp.get('department', 'Unknown')
                    context_parts.append(
                        f"{i}. **{emp_name}** ({dept}): {risk_score:.1f}% risk\n"
                    )
            
            # Cache the result
            self._cached_context = "".join(context_parts)
            self._context_hash = current_hash
            
            return self._cached_context
            
        except Exception as e:
            return f"Error building data context: {e}"
    
    def _search_employee(self, search_term: str) -> Optional[Dict[str, Any]]:
        """
        Search for an employee by name or ID.
        
        Args:
            search_term: Employee name or ID to search for
            
        Returns:
            Employee prediction dictionary if found, None otherwise
        """
        if not search_term or not search_term.strip():
            return None
        
        search_term = search_term.strip()
        search_term_lower = search_term.lower()
        
        try:
            # Try exact ID match first (case-insensitive)
            if 'Employee_ID' in self.df.columns:
                id_match = self.df[self.df['Employee_ID'].astype(str).str.lower() == search_term_lower]
                if not id_match.empty:
                    return predict_attrition_risk(id_match.iloc[0])
            
            # Try exact name match (case-insensitive)
            if 'Name' in self.df.columns:
                name_match = self.df[self.df['Name'].astype(str).str.lower() == search_term_lower]
                if not name_match.empty:
                    return predict_attrition_risk(name_match.iloc[0])
            
            # Try partial name match (case-insensitive)
            if 'Name' in self.df.columns:
                partial_match = self.df[self.df['Name'].astype(str).str.lower().str.contains(search_term_lower, na=False)]
                if not partial_match.empty:
                    # If multiple matches, return the first one
                    return predict_attrition_risk(partial_match.iloc[0])
            
            # Try partial ID match (case-insensitive)
            if 'Employee_ID' in self.df.columns:
                partial_id_match = self.df[self.df['Employee_ID'].astype(str).str.lower().str.contains(search_term_lower, na=False)]
                if not partial_id_match.empty:
                    return predict_attrition_risk(partial_id_match.iloc[0])
            
        except Exception as e:
            print(f"Error searching for employee: {e}")
            return None
        
        return None
    
    def _detect_employee_query(self, query: str) -> Optional[str]:
        """
        Detect if query mentions a specific employee name or ID.
        
        Returns:
            Search term (name or ID) if detected, None otherwise
        """
        if not query:
            return None
        
        query_lower = query.lower()
        
        # Check for employee ID patterns (EMP001, EMP-001, etc.)
        id_patterns = [
            r'\b(EMP\d+)\b',  # EMP001, EMP123
            r'\b(EMP-\d+)\b',  # EMP-001
            r'\b(EMP_\d+)\b',  # EMP_001
        ]
        
        for pattern in id_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                return matches[0]  # Return first match
        
        # Try to extract employee name
        # Common patterns: "about [Name]", "for [Name]", "[Name]'s risk", etc.
        
        # Skip common query words
        skip_words = {'tell', 'me', 'about', 'what', 'are', 'the', 'for', 'of', 
                     'show', 'give', 'provide', 'how', 'can', 'we', 'prevent',
                     'stop', 'help', 'employee', 'risk', 'drivers', 'recommendations',
                     'suggestions', 'advice', 'action', 'should', 'do', 'to', 'their',
                     'his', 'her', 'from', 'leaving', 'quit', 'resign'}
        
        words = query.split()
        potential_names = []
        
        # Try to find quoted strings first (most likely to be names)
        quoted_pattern = r'["\']([^"\']+)["\']'
        quoted_matches = re.findall(quoted_pattern, query)
        for match in quoted_matches:
            match_clean = match.strip('.,!?;:')
            if len(match_clean) > 1 and match_clean.lower() not in skip_words:
                potential_names.append(match_clean)
        
        # Extract capitalized words (potential names)
        # Group consecutive capitalized words as potential multi-word names
        name_parts = []
        for i, word in enumerate(words):
            word_clean = word.strip('.,!?;:')
            word_lower = word_clean.lower()
            
            if word_lower in skip_words:
                # If we hit a stop word, finalize current name if any
                if name_parts:
                    potential_name = ' '.join(name_parts)
                    if len(potential_name) > 1:
                        potential_names.append(potential_name)
                    name_parts = []
                continue
            
            # Check if word starts with capital (potential name part)
            if word_clean and word_clean[0].isupper() and len(word_clean) > 1:
                name_parts.append(word_clean)
            else:
                # If we hit a non-capitalized word, finalize current name if any
                if name_parts:
                    potential_name = ' '.join(name_parts)
                    if len(potential_name) > 1:
                        potential_names.append(potential_name)
                    name_parts = []
        
        # Don't forget the last name if we're at the end
        if name_parts:
            potential_name = ' '.join(name_parts)
            if len(potential_name) > 1:
                potential_names.append(potential_name)
        
        # Try searching with potential names (longest first for more specific matches)
        if potential_names:
            # Sort by length (longest first) and remove duplicates while preserving order
            seen = set()
            unique_names = []
            for name in sorted(potential_names, key=len, reverse=True):
                if name not in seen:
                    seen.add(name)
                    unique_names.append(name)
            
            for name in unique_names:
                # Try searching with this name
                result = self._search_employee(name)
                if result:
                    return name
        
        return None
    
    def _format_employee_context(self, employee_pred: Dict[str, Any]) -> str:
        """
        Format detailed employee prediction data for LLM context.
        
        Includes:
        - Basic info (name, ID, department, level, tenure)
        - Risk assessment (score, level, trend)
        - Top 3 risk drivers with descriptions and benchmarks
        - Key metrics (performance, engagement trends)
        - Qualitative data (manager notes, survey comments)
        """
        context_parts = [
            "## Specific Employee Analysis Requested\n\n",
            f"**Employee**: {employee_pred.get('employee_name', 'Unknown')} (ID: {employee_pred.get('employee_id', 'Unknown')})\n",
            f"**Department**: {employee_pred.get('department', 'Unknown')} | **Level**: {employee_pred.get('level', 'Unknown')} | **Tenure**: {employee_pred.get('tenure_years', 0):.1f} years\n",
            f"**Manager**: {employee_pred.get('manager_name', 'N/A')}\n\n"
        ]
        
        # Risk Assessment
        context_parts.append("### Risk Assessment\n")
        context_parts.append(f"- **Risk Score**: {employee_pred.get('risk_score', 0):.1f}% ({employee_pred.get('risk_level', 'Unknown')})\n")
        context_parts.append(f"- **Risk Trend**: {employee_pred.get('risk_trend', 'Unknown')} {employee_pred.get('risk_trend_icon', '')}\n\n")
        
        # Top Risk Drivers
        top_drivers = employee_pred.get('top_3_drivers', [])
        if top_drivers:
            context_parts.append("### Top Risk Drivers\n")
            for i, driver in enumerate(top_drivers, 1):
                context_parts.append(f"{i}. **{driver.get('name', 'Unknown')}** ({driver.get('influence_pct', 0)}% influence)\n")
                context_parts.append(f"   - Description: {driver.get('description', 'N/A')}\n")
                context_parts.append(f"   - Benchmark: {driver.get('benchmark', 'N/A')}\n\n")
        else:
            context_parts.append("### Top Risk Drivers\n")
            context_parts.append("No significant risk drivers detected. Employee profile appears stable.\n\n")
        
        # Key Metrics
        features = employee_pred.get('features', {})
        if features:
            context_parts.append("### Key Metrics\n")
            
            # Performance metrics
            perf_current = features.get('Performance_Current')
            perf_trend = features.get('Performance_Trend', 0)
            if perf_current is not None:
                trend_str = "increasing" if perf_trend > 0 else "decreasing" if perf_trend < 0 else "stable"
                context_parts.append(f"- **Performance**: {perf_current:.1f} (Trend: {trend_str})\n")
            
            # Engagement metrics
            eng_current = features.get('Engagement_Current')
            eng_trend = features.get('Engagement_Trend', 0)
            if eng_current is not None:
                trend_str = "increasing" if eng_trend > 0 else "decreasing" if eng_trend < 0 else "stable"
                context_parts.append(f"- **Engagement**: {eng_current:.1f}/10 (Trend: {trend_str})\n")
            
            # Career progression
            months_since_promotion = features.get('Months_Since_Promotion')
            if months_since_promotion is not None:
                context_parts.append(f"- **Months Since Promotion**: {months_since_promotion:.0f}\n")
            
            # Compensation gap
            comp_gap = features.get('Compensation_Gap_Pct')
            if comp_gap is not None:
                context_parts.append(f"- **Compensation Gap**: {comp_gap:.1f}% vs market average\n")
            
            context_parts.append("\n")
        
        # Qualitative Feedback
        manager_notes = employee_pred.get('manager_notes', '')
        survey_comments = employee_pred.get('survey_comments', '')
        
        if manager_notes or survey_comments:
            context_parts.append("### Qualitative Feedback\n")
            if manager_notes:
                context_parts.append(f"- **Manager Notes**: {manager_notes}\n")
            if survey_comments:
                context_parts.append(f"- **Survey Comments**: {survey_comments}\n")
            context_parts.append("\n")
        
        context_parts.append("---\n\n")
        context_parts.append("When providing recommendations for this employee, focus on addressing the top risk drivers listed above with specific, actionable steps.")
        
        return "".join(context_parts)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return """You are an expert HR Analytics Assistant helping executives analyze employee attrition risk data.

Your capabilities:
- Analyze employee risk patterns and trends
- Provide insights on department-level risk
- Recommend retention strategies
- Answer questions about specific employees or departments
- Explain risk drivers and their implications
- Provide personalized, actionable recommendations for preventing employee attrition

Guidelines:
- Be conversational, professional, and helpful
- Use markdown formatting for clarity (headers, lists, bold text)
- Provide specific numbers and percentages when available
- Offer actionable recommendations when appropriate
- If asked about specific employees, use the provided data context (especially detailed employee analysis if provided)
- Be concise but thorough

When providing recommendations for specific employees:
- Focus on addressing the top risk drivers identified for that employee
- Provide specific, actionable steps (not vague advice)
- Prioritize recommendations (most important first)
- Suggest timelines when appropriate (immediate actions vs. short-term vs. long-term)
- Consider the employee's specific situation (department, level, tenure, etc.)
- Reference specific risk drivers and their influence percentages

You have access to real-time employee risk data that will be provided in the conversation context."""
    
    def _prepare_messages(self, query: str, conversation_history: List[Dict] = None) -> List[Dict]:
        """Prepare messages for API call."""
        # Detect if query is about a specific employee
        employee_search_term = self._detect_employee_query(query)
        employee_context = None
        
        if employee_search_term:
            # Search for the employee
            employee_pred = self._search_employee(employee_search_term)
            if employee_pred:
                # Format detailed employee data
                employee_context = self._format_employee_context(employee_pred)
        
        # Build general data context (uses cache)
        data_context = self._build_data_context()
        
        # Format conversation history
        messages = [
            {"role": "system", "content": self._get_system_prompt()}
        ]
        
        # Build context string - employee-specific data first if available
        context_parts = []
        if employee_context:
            context_parts.append(employee_context)
            context_parts.append("\n\n")
        
        context_parts.append("Current general data context:\n\n")
        context_parts.append(data_context)
        context_parts.append("\n\nUse this data to answer questions accurately.")
        
        # Add data context as system message
        messages.append({
            "role": "system",
            "content": "".join(context_parts)
        })
        
        # Add conversation history (last 10 messages to manage token limits)
        if conversation_history:
            recent_history = conversation_history[-10:]
            for msg in recent_history:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role in ['user', 'assistant'] and content:
                    messages.append({"role": role, "content": content})
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def process_query_stream(self, query: str, ui_context: Dict[str, Any] = None, 
                            conversation_history: List[Dict] = None) -> Iterator[str]:
        """
        Process user query with streaming response.
        
        Args:
            query: User's query
            ui_context: UI context (optional)
            conversation_history: Previous messages
            
        Yields:
            Text chunks as they arrive from the API
        """
        if not query or not query.strip():
            yield "Hello! I'm your HR Analytics Assistant. How can I help you today?"
            return
        
        # If no LLM client available, return fallback response
        if not self.client:
            yield "I'm currently unavailable. Please check API configuration."
            return
        
        try:
            # Prepare messages
            messages = self._prepare_messages(query, conversation_history)
            
            # Call Groq API with streaming
            stream = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                stream=True
            )
            
            # Yield chunks as they arrive
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            # Error handling
            yield f"I encountered an error processing your request: {str(e)}"
    
    def process_query(self, query: str, ui_context: Dict[str, Any] = None, 
                     conversation_history: List[Dict] = None, stream: bool = False) -> Dict[str, Any]:
        """
        Process user query using LLM with data context.
        
        Args:
            query: User's query
            ui_context: UI context (optional)
            conversation_history: Previous messages
            stream: Whether to use streaming (returns generator)
            
        Returns:
            Response dictionary with type, content, and optional data
            If stream=True, returns generator instead
        """
        if stream:
            # Return generator for streaming
            return {
                'type': 'stream',
                'stream': self.process_query_stream(query, ui_context, conversation_history)
            }
        
        if not query or not query.strip():
            return {
                'type': 'text',
                'content': "Hello! I'm your HR Analytics Assistant. How can I help you today?"
            }
        
        # If no LLM client available, return fallback response
        if not self.client:
            return {
                'type': 'text',
                'content': "I'm currently unavailable. Please check API configuration."
            }
        
        try:
            # Prepare messages
            messages = self._prepare_messages(query, conversation_history)
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract response
            llm_response = response.choices[0].message.content
            
            return {
                'type': 'text',
                'content': llm_response
            }
            
        except Exception as e:
            # Error handling - return helpful error message
            error_msg = f"I encountered an error processing your request: {str(e)}"
            return {
                'type': 'text',
                'content': error_msg
            }

