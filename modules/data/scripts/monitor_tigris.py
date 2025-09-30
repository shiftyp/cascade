#!/usr/bin/env python3
"""
Monitor Tigris storage usage and costs for CASCADE Data Collector

Shows current usage, growth trends, and cost projections.
"""

import os
import sys
import asyncio
import aioboto3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

class TigrisMonitor:
    """Monitor Tigris storage usage and costs."""

    def __init__(self):
        self.endpoint = os.getenv('AWS_ENDPOINT_URL_S3', 'https://fly.storage.tigris.dev')
        self.access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')

        # Buckets to monitor
        self.buckets = {
            'main': 'cascade-iq-data',
            'qa': 'cascade-qa-samples'
        }

        # Pricing (Tigris)
        self.cost_per_gb_month = 0.02
        self.cost_per_1000_put = 0.005
        self.cost_per_1000_get = 0.0004

    async def get_bucket_stats(self, bucket_name: str) -> Dict:
        """Get statistics for a bucket."""
        session = aioboto3.Session()

        async with session.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name='auto'
        ) as s3:

            total_size = 0
            total_objects = 0

            # List and sum all objects
            paginator = s3.get_paginator('list_objects_v2')

            async for page in paginator.paginate(Bucket=bucket_name):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        total_size += obj['Size']
                        total_objects += 1

            # Get bucket creation date for age calculation
            try:
                bucket_info = await s3.head_bucket(Bucket=bucket_name)
                creation_date = datetime.now() - timedelta(days=365)  # Estimate
            except:
                creation_date = datetime.now()

            return {
                'name': bucket_name,
                'total_size_bytes': total_size,
                'total_size_gb': total_size / (1024**3),
                'total_size_tb': total_size / (1024**4),
                'object_count': total_objects,
                'avg_object_size_mb': (total_size / total_objects / (1024**2)) if total_objects > 0 else 0,
                'monthly_storage_cost': (total_size / (1024**3)) * self.cost_per_gb_month,
                'creation_date': creation_date.isoformat()
            }

    async def get_growth_projection(self, current_gb: float, months_elapsed: int) -> Dict:
        """Project future growth based on current usage."""
        if months_elapsed == 0:
            growth_rate_gb_per_month = 50  # Initial estimate
        else:
            growth_rate_gb_per_month = current_gb / months_elapsed

        projections = {}
        for month in [1, 3, 6, 12, 18]:
            projected_gb = current_gb + (growth_rate_gb_per_month * month)
            projections[f"month_{month}"] = {
                'size_gb': projected_gb,
                'size_tb': projected_gb / 1024,
                'cost': projected_gb * self.cost_per_gb_month
            }

        return {
            'current_growth_rate_gb_month': growth_rate_gb_per_month,
            'projections': projections
        }

    async def analyze_qa_samples(self, bucket_name: str) -> Dict:
        """Analyze QA sample distribution and diversity."""
        session = aioboto3.Session()

        async with session.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name='auto'
        ) as s3:

            # Sample recent QA files for analysis
            recent_samples = []
            paginator = s3.get_paginator('list_objects_v2')

            async for page in paginator.paginate(
                Bucket=bucket_name,
                Prefix='qa_samples/',
                MaxKeys=100
            ):
                if 'Contents' in page:
                    recent_samples.extend(page['Contents'][:10])

            # Analyze selection reasons from metadata
            selection_reasons = {'random': 0, 'diversity': 0, 'transition': 0, 'anomaly': 0}

            for sample in recent_samples:
                try:
                    # Get object metadata
                    obj = await s3.head_object(Bucket=bucket_name, Key=sample['Key'])
                    metadata = obj.get('Metadata', {})
                    reason = metadata.get('selection_reason', 'unknown')
                    if reason in selection_reasons:
                        selection_reasons[reason] += 1
                except:
                    pass

            return {
                'total_analyzed': len(recent_samples),
                'selection_distribution': selection_reasons,
                'diversity_ratio': selection_reasons['diversity'] / max(1, sum(selection_reasons.values()))
            }

    async def generate_report(self):
        """Generate comprehensive storage report."""
        print("=" * 60)
        print("CASCADE TIGRIS STORAGE MONITOR")
        print("=" * 60)
        print(f"Report Generated: {datetime.now().isoformat()}")
        print()

        total_cost = 0
        total_size_tb = 0

        # Monitor each bucket
        for bucket_type, bucket_name in self.buckets.items():
            try:
                stats = await self.get_bucket_stats(bucket_name)

                print(f"📦 {bucket_type.upper()} BUCKET: {bucket_name}")
                print(f"   Size: {stats['total_size_gb']:.2f} GB ({stats['total_size_tb']:.3f} TB)")
                print(f"   Objects: {stats['object_count']:,}")
                print(f"   Avg Size: {stats['avg_object_size_mb']:.2f} MB")
                print(f"   Monthly Cost: ${stats['monthly_storage_cost']:.2f}")
                print()

                total_cost += stats['monthly_storage_cost']
                total_size_tb += stats['total_size_tb']

                # QA sample analysis
                if bucket_type == 'qa':
                    qa_analysis = await self.analyze_qa_samples(bucket_name)
                    print(f"   📊 QA Sample Analysis:")
                    print(f"      Diversity Ratio: {qa_analysis['diversity_ratio']*100:.1f}%")
                    print(f"      Selection Methods: {qa_analysis['selection_distribution']}")
                    print()

            except Exception as e:
                print(f"   ⚠️  Error accessing bucket: {e}")
                print()

        # Growth projections
        print("📈 GROWTH PROJECTIONS:")
        projection = await self.get_growth_projection(total_size_tb * 1024, 1)  # Convert TB to GB

        for period, data in projection['projections'].items():
            print(f"   {period}: {data['size_tb']:.2f} TB (${data['cost']:.2f}/month)")
        print()

        # Cost summary
        print("💰 COST SUMMARY:")
        print(f"   Current Storage: ${total_cost:.2f}/month")
        print(f"   Egress Fees: $0.00 (FREE with Tigris!)")
        print(f"   Total: ${total_cost:.2f}/month")
        print()

        # Comparison with alternatives
        print("📊 COST COMPARISON:")
        aws_s3_cost = total_size_tb * 1024 * 0.023  # S3 Standard
        aws_egress = total_size_tb * 1024 * 0.09 * 0.1  # Assume 10% monthly egress

        print(f"   Tigris: ${total_cost:.2f}/month")
        print(f"   AWS S3 Standard: ${aws_s3_cost:.2f}/month + ${aws_egress:.2f} egress")
        print(f"   Savings: ${(aws_s3_cost + aws_egress - total_cost):.2f}/month")
        print()

        # Recommendations
        print("💡 RECOMMENDATIONS:")
        if total_size_tb > 30:
            print("   ⚠️  Consider hybrid approach with S3 Glacier for old data")
            print("      Potential savings: $760/month for 40TB archive")

        if projection['current_growth_rate_gb_month'] > 3000:
            print("   ⚠️  High growth rate detected")
            print("      Consider increasing compression or sampling rates")

        if total_cost < 100:
            print("   ✅ Storage costs well within budget")
            print("   ✅ Continue with current Tigris-only strategy")

        print()
        print("=" * 60)

async def main():
    """Run the monitor."""
    monitor = TigrisMonitor()

    # Check if credentials are configured
    if not monitor.access_key:
        print("❌ AWS_ACCESS_KEY_ID not set")
        print("   Run: fly ssh console -C 'env | grep AWS_'")
        print("   Or: fly storage list")
        sys.exit(1)

    await monitor.generate_report()

if __name__ == "__main__":
    asyncio.run(main())