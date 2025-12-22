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
    get_high_risk_employees
)


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
                os.path.dirname(os.path.dirname(__file__)),
                "Daily news summary",
                "production",
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
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return """You are an expert HR Analytics Assistant helping executives analyze employee attrition risk data.

Your capabilities:
- Analyze employee risk patterns and trends
- Provide insights on department-level risk
- Recommend retention strategies
- Answer questions about specific employees or departments
- Explain risk drivers and their implications

Guidelines:
- Be conversational, professional, and helpful
- Use markdown formatting for clarity (headers, lists, bold text)
- Provide specific numbers and percentages when available
- Offer actionable recommendations when appropriate
- If asked about specific employees, use the provided data context
- Be concise but thorough

You have access to real-time employee risk data that will be provided in the conversation context."""
    
    def _prepare_messages(self, query: str, conversation_history: List[Dict] = None) -> List[Dict]:
        """Prepare messages for API call."""
        # Build data context (uses cache)
        data_context = self._build_data_context()
        
        # Format conversation history
        messages = [
            {"role": "system", "content": self._get_system_prompt()}
        ]
        
        # Add data context as system message
        messages.append({
            "role": "system",
            "content": f"Current data context:\n\n{data_context}\n\nUse this data to answer questions accurately."
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

