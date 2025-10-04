#!/usr/bin/env python3
"""
Test specific syllabus-based queries
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

def test_specific_queries():
    """Test specific syllabus queries"""
    
    queries = [
        "What should I study for DSA first test?",
        "Help me prepare for Database exam",
        "Machine Learning topics for second semester",
        "Database preparation guide",
        "ML syllabus second semester"
    ]
    
    print("🎓 Testing Syllabus-Based Queries...")
    print("=" * 60)
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 40)
        
        try:
            response = rag.answer(query)
            answer = response.get('answer', '')
            sources = response.get('sources', [])
            
            print(f"Answer: {answer[:300]}...")
            print(f"Sources: {len(sources)} found")
            
            # Check if it's a syllabus-based response
            if any(keyword in answer.lower() for keyword in ['unit 1', 'unit 2', 'syllabus', 'semester']):
                print("✅ Syllabus-based response detected!")
            else:
                print("⚠️  Generic response - syllabus integration may need improvement")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎓 Syllabus query test completed!")

if __name__ == "__main__":
    test_specific_queries()