# Update minimum redundancy validation
parser.add_argument(
    '--redundancy',
    type=int,
    choices=[1, 2, 4, 6],  # Add 1 to allowed values
    default=2,
    help='Pattern redundancy factor (1=no redundancy, 2/4/6=with redundancy)'
)

# ...existing code...

def validate_args(args):
    """Validate command line arguments."""
    if args.redundancy not in [1, 2, 4, 6]:
        raise ValueError("Redundancy must be one of: 1, 2, 4, 6")
    # ...existing code...

# ...existing code...

def generate_pattern_with_redundancy(base_pattern, redundancy):
    """
    Generate pattern with optional redundancy.
    If redundancy=1, return base pattern as-is (no redundancy).
    """
    if redundancy == 1:
        # No redundancy - return pure orthogonal pattern
        return base_pattern
    
    # Apply redundancy spreading for redundancy > 1
    # ...existing code...

def apply_redundancy(pattern, redundancy):
    """Apply redundancy to pattern. If redundancy=1, return as-is."""
    if redundancy == 1:
        # No redundancy - pure orthogonal pattern
        return pattern
    
    # Apply spreading/repetition for redundancy > 1
    # ...existing code...