#!/usr/bin/env python3
"""
Demo of Enhanced Study Buddy with MCA Syllabus Integration
"""

import os
import sys
import django

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from chat.rag import rag

def demo_study_buddy():
    """Demonstrate the enhanced study buddy capabilities"""
    
    print("🎓 CMR University Study Buddy - Enhanced with MCA Syllabus Integration")
    print("=" * 80)
    
    demo_queries = [
        {
            "category": "🎯 Syllabus-Based Test Preparation",
            "queries": [
                "What should I study for DSA first test?",
                "Help me prepare for Database exam",
                "Machine Learning topics for second semester"
            ]
        },
        {
            "category": "💪 Academic Support & Motivation",
            "queries": [
                "I'm struggling with my studies",
                "Give me some study tips",
                "I need motivation for exams"
            ]
        },
        {
            "category": "🏫 University Information",
            "queries": [
                "Tell me about campus life",
                "What about placements?",
                "How do I apply for MCA?"
            ]
        },
        {
            "category": "🤖 Study Buddy Features",
            "queries": [
                "Who are you?",
                "What can you help me with?"
            ]
        }
    ]
    
    for category_info in demo_queries:
        print(f"\n{category_info['category']}")
        print("-" * 60)
        
        for query in category_info['queries']:
            print(f"\n📝 Student: \"{query}\"")
            print("🤖 Study Buddy:")
            
            try:
                response = rag.answer(query)
                answer = response.get('answer', '')
                sources = response.get('sources', [])
                
                # Show first 400 characters of response
                if len(answer) > 400:
                    print(f"{answer[:400]}...")
                else:
                    print(answer)
                
                if sources:
                    print(f"📚 Sources: {len(sources)} found")
                
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print()
    
    print("=" * 80)
    print("✨ The Study Buddy now provides:")
    print("• Syllabus-based study guidance for all MCA subjects")
    print("• Test preparation with specific unit breakdowns")
    print("• Motivational support and study tips")
    print("• University information and career guidance")
    print("• 24/7 friendly academic companion")
    print("\n🎓 Your success is our mission! Ask me anything about CMR University or your studies!")

if __name__ == "__main__":
    demo_study_buddy()