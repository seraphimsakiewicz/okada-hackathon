#!/usr/bin/env python3
"""Comprehensive automated test for all properties in the RAG dataset"""

import pandas as pd
import requests
import json
import time
import sys
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PropertyTester:
    def __init__(self, base_url: str = "http://localhost:8000", delay: float = 1.0):
        self.base_url = base_url
        self.delay = delay  # Delay between requests to be respectful
        self.results = []
        
    def load_properties(self, csv_path: str) -> pd.DataFrame:
        """Load properties from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} properties from {csv_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def construct_query(self, row: pd.Series) -> str:
        """Construct a query for testing a specific property"""
        address = row['Property Address']
        floor = row['Floor'] 
        suite = row['Suite']
        
        # Handle different floor/suite formats
        if pd.notna(floor) and pd.notna(suite):
            return f"Who are the associates that manage the property on {address}, Suite {suite}, Floor {floor}?"
        elif pd.notna(floor):
            return f"Who are the associates that manage the property on {address} on floor {floor}?"
        else:
            return f"Who are the associates that manage the property on {address}?"
    
    def extract_expected_associates(self, row: pd.Series) -> List[str]:
        """Extract expected associates from the CSV row"""
        associates = []
        for i in range(1, 5):  # Associate 1-4
            col_name = f'Associate {i}'
            if col_name in row and pd.notna(row[col_name]):
                associates.append(row[col_name].strip())
        return associates
    
    def send_chat_request(self, query: str, conversation_id: str) -> Dict[str, Any]:
        """Send a chat request to the API"""
        try:
            payload = {
                "conversation_id": conversation_id,
                "message": query
            }
            
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                return {"error": f"API error {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Request error: {e}")
            return {"error": str(e)}
    
    def validate_response(self, response_text: str, expected_associates: List[str]) -> Dict[str, Any]:
        """Validate that the response contains the expected associates"""
        response_lower = response_text.lower()
        
        found_associates = []
        missing_associates = []
        
        for associate in expected_associates:
            if associate.lower() in response_lower:
                found_associates.append(associate)
            else:
                missing_associates.append(associate)
        
        success = len(missing_associates) == 0
        accuracy = len(found_associates) / len(expected_associates) if expected_associates else 0
        
        return {
            "success": success,
            "accuracy": accuracy,
            "found_associates": found_associates,
            "missing_associates": missing_associates,
            "total_expected": len(expected_associates)
        }
    
    def test_property(self, row: pd.Series, property_id: int) -> Dict[str, Any]:
        """Test a single property"""
        query = self.construct_query(row)
        expected_associates = self.extract_expected_associates(row)
        conversation_id = f"test_property_{property_id}"
        
        logger.info(f"Testing Property {property_id}: {row['Property Address']} - {query[:60]}...")
        
        # Send API request
        start_time = time.time()
        response = self.send_chat_request(query, conversation_id)
        request_time = time.time() - start_time
        
        # Check for API errors
        if "error" in response:
            return {
                "property_id": property_id,
                "address": row['Property Address'],
                "floor": row.get('Floor', ''),
                "suite": row.get('Suite', ''),
                "query": query,
                "expected_associates": expected_associates,
                "success": False,
                "error": response["error"],
                "request_time": request_time
            }
        
        # Validate response
        validation = self.validate_response(response.get("response", ""), expected_associates)
        
        result = {
            "property_id": property_id,
            "address": row['Property Address'],
            "floor": row.get('Floor', ''),
            "suite": row.get('Suite', ''),
            "query": query,
            "expected_associates": expected_associates,
            "response": response.get("response", ""),
            "rag_used": response.get("rag_context_used", False),
            "processing_time": response.get("processing_time", 0),
            "request_time": request_time,
            **validation
        }
        
        return result
    
    def run_comprehensive_test(self, csv_path: str, limit: int = None, start: int = None) -> Dict[str, Any]:
        """Run comprehensive test on all properties"""
        logger.info("🚀 Starting comprehensive property test...")
        
        # Load properties
        df = self.load_properties(csv_path)
        
        # Apply start and limit for testing (optional)
        if start is not None:
            df = df.iloc[start:]
            logger.info(f"Starting from property {start + 1}")
        
        if limit:
            df = df.head(limit)
            logger.info(f"Testing {limit} properties")
            
        if start is not None:
            logger.info(f"Testing properties {start + 1} to {start + len(df)}")
        
        # Test each property
        total_properties = len(df)
        successful_tests = 0
        total_accuracy = 0
        
        for idx, row in df.iterrows():
            result = self.test_property(row, idx + 1)
            self.results.append(result)
            
            if result["success"]:
                successful_tests += 1
            
            total_accuracy += result.get("accuracy", 0)
            
            # Progress update
            if (idx + 1) % 10 == 0:
                progress = (idx + 1) / total_properties * 100
                logger.info(f"Progress: {idx + 1}/{total_properties} ({progress:.1f}%) - Success rate: {successful_tests/(idx+1)*100:.1f}%")
            
            # Rate limiting
            time.sleep(self.delay)
        
        # Generate summary
        avg_accuracy = total_accuracy / total_properties
        success_rate = successful_tests / total_properties * 100
        
        summary = {
            "total_properties": total_properties,
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "average_accuracy": avg_accuracy,
            "failed_tests": total_properties - successful_tests
        }
        
        return summary
    
    def generate_report(self, summary: Dict[str, Any]) -> None:
        """Generate detailed test report"""
        print("\n" + "="*80)
        print("🎯 COMPREHENSIVE PROPERTY TEST REPORT")
        print("="*80)
        
        print(f"📊 OVERALL RESULTS:")
        print(f"   Total Properties Tested: {summary['total_properties']}")
        print(f"   Successful Tests: {summary['successful_tests']}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        print(f"   Average Accuracy: {summary['average_accuracy']*100:.1f}%")
        print(f"   Failed Tests: {summary['failed_tests']}")
        
        # Failed tests details
        failed_tests = [r for r in self.results if not r["success"]]
        if failed_tests:
            print(f"\n❌ FAILED TESTS ({len(failed_tests)}):")
            for test in failed_tests:
                print(f"   • Property {test['property_id']}: {test['address']}")
                if "error" in test:
                    print(f"     Error: {test['error']}")
                else:
                    print(f"     Missing: {test.get('missing_associates', [])}")
        
        # Performance stats
        if self.results:
            avg_processing_time = sum(r.get('processing_time', 0) for r in self.results) / len(self.results)
            avg_request_time = sum(r.get('request_time', 0) for r in self.results) / len(self.results)
            
            print(f"\n⚡ PERFORMANCE:")
            print(f"   Average Processing Time: {avg_processing_time:.2f}s")
            print(f"   Average Request Time: {avg_request_time:.2f}s")
        
        print("="*80)

def main():
    """Main function to run the comprehensive test"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive property test')
    parser.add_argument('--csv', default='rag_data/HackathonInternalKnowledgeBase.csv', help='Path to CSV file')
    parser.add_argument('--limit', type=int, help='Limit number of properties to test (for quick testing)')
    parser.add_argument('--start', type=int, help='Starting property index (0-based)')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between requests in seconds')
    parser.add_argument('--url', default='http://localhost:8000', help='Base URL for API')
    
    args = parser.parse_args()
    
    # Create tester
    tester = PropertyTester(base_url=args.url, delay=args.delay)
    
    # Run test
    try:
        summary = tester.run_comprehensive_test(args.csv, limit=args.limit, start=args.start)
        tester.generate_report(summary)
        
        # Save detailed results
        import json
        with open('property_test_results.json', 'w') as f:
            json.dump({
                'summary': summary,
                'detailed_results': tester.results
            }, f, indent=2, default=str)
        
        logger.info("📁 Detailed results saved to property_test_results.json")
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        if tester.results:
            logger.info(f"Partial results available for {len(tester.results)} properties")

if __name__ == "__main__":
    main()