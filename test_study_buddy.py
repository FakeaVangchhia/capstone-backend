#!/usr/bin/env python3
"""
Simple test script to verify the Study Buddy functionality
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

def test_study_buddy_responses():
    """Test various study buddy interactions"""
    
    test_cases = [
        {
            "query": "Hello",
            "expected_keywords": ["Study Buddy", "CMR University", "help"]
        },
        {
            "query": "Who are you?",
            "expected_keywords": ["Study Buddy", "CMR University", "academic guidance"]
        },
        {
            "query": "I need motivation",
            "expected_keywords": ["motivation", "you've got this", "support"]
        },
        {
            "query": "Study tips please",
            "expected_keywords": ["Tip of the Day", "study"]
        },
        {
            "query": "Tell me about admissions",
            "expected_keywords": ["admission", "process", "application"]
        },
        {
            "query": "Campus life information",
            "expected_keywords": ["campus", "hostel", "activities"]
        },
        {
            "query": "Placement information",
            "expected_keywords": ["placement", "career", "companies"]
        },
        {
            "query": "What should I study for DSA first test?",
            "expected_keywords": ["Unit 1", "Unit 2", "data structures", "hashing"]
        },
        {
            "query": "Help me prepare for Database exam",
            "expected_keywords": ["relational", "SQL", "normalization", "database"]
        },
        {
            "query": "Machine Learning topics for second semester",
            "expected_keywords": ["supervised", "unsupervised", "neural networks", "python"]
        }
    ]
    
    print("🤖 Testing CMR University Study Buddy...")
    print("=" * 50)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: '{test['query']}'")
        print("-" * 30)
        
        try:
            response = rag.answer(test['query'])
            answer = response.get('answer', '')
            
            print(f"Response: {answer[:200]}...")
            
            # Check if expected keywords are present (case insensitive)
            found_keywords = []
            for keyword in test['expected_keywords']:
                if keyword.lower() in answer.lower():
                    found_keywords.append(keyword)
            
            if found_keywords:
                print(f"✅ Found keywords: {found_keywords}")
            else:
                print(f"⚠️  Expected keywords not found: {test['expected_keywords']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎓 Study Buddy test completed!")

def test_motivation_and_tips():
    """Test the motivation and study tip functions"""
    print("\n🌟 Testing Motivation & Study Tips...")
    print("=" * 50)
    
    try:
        # Test motivation
        motivation = rag.get_study_motivation()
        print(f"Motivation: {motivation}")
        
        # Test study tip
        tip = rag.get_study_tip_of_day()
        print(f"Study Tip: {tip}")
        
        print("✅ Motivation and tips working correctly!")
        
    except Exception as e:
        print(f"❌ Error testing motivation/tips: {e}")

if __name__ == "__main__":
    test_study_buddy_responses()
    test_motivation_and_tips()