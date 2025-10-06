# Update minimum redundancy validation
parser.add_argument(
    '--redundancy',
    type=int,
    default=2,
    help='Pattern redundancy factor (minimum 1, no built-in redundancy at 1x)'
)

# ...existing code...

def validate_args(args):
    """Validate command line arguments."""
    if args.redundancy < 1:
        raise ValueError("Redundancy must be at least 1")
    # ...existing code...

# ...existing code...

def generate_pattern_with_redundancy(base_pattern, redundancy):
    """
    Generate pattern with redundancy spreading.
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