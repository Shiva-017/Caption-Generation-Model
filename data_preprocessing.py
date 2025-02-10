def sanitize_flicker30k_captions(captions_file, images_dir, output_captions_file):
    """
    Sanitize Flickr30k captions by expanding the captions list and ensuring images exist.
    """
    # Try reading the CSV file with different separators
    try:
        df = pd.read_csv(captions_file, sep='\t', engine='python')
    except Exception as e:
        logging.error(f"Error reading CSV with tab separator: {e}")
        df = pd.read_csv(captions_file, engine='python')
    
    # Print columns to debug
    print("Columns in DataFrame:", df.columns.tolist())
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    print("Columns after stripping whitespace:", df.columns.tolist())
    
    # Check if 'filename' column exists
    if 'filename' not in df.columns:
        logging.error("Column 'filename' not found in CSV file.")
        logging.info(f"Available columns: {df.columns.tolist()}")
        return None
    
    logging.info(f"Original number of entries: {len(df)}")
    
    # Proceed with processing
    records = []
    for idx, row in df.iterrows():
        filename = row['filename']
        split = row['split']
        try:
            captions_list = ast.literal_eval(row['raw'])
        except (ValueError, SyntaxError):
            logging.warning(f"Invalid format in 'raw' column at index {idx}")
            continue
        for caption in captions_list:
            records.append({
                'image': filename,
                'caption': caption.strip(),
                'split': split
            })
    
    expanded_df = pd.DataFrame(records)
    logging.info(f"Expanded to {len(expanded_df)} image-caption pairs")
    
    # Remove entries with missing captions
    expanded_df.dropna(subset=['caption'], inplace=True)
    logging.info(f"After removing missing captions: {len(expanded_df)}")
    
    # Ensure all images exist
    expanded_df['image_path'] = expanded_df['image'].apply(lambda x: os.path.join(images_dir, x))
    expanded_df['image_exists'] = expanded_df['image_path'].apply(os.path.exists)
    missing_images = expanded_df[~expanded_df['image_exists']]
    if not missing_images.empty:
        logging.warning(f"Number of missing images: {len(missing_images)}")
        expanded_df = expanded_df[expanded_df['image_exists']]
    else:
        logging.info("All images are present.")
    
    # Remove duplicates
    initial_len = len(expanded_df)
    expanded_df.drop_duplicates(subset=['image', 'caption'], inplace=True)
    logging.info(f"Removed {initial_len - len(expanded_df)} duplicate caption entries.")
    
    # Reset index
    expanded_df.reset_index(drop=True, inplace=True)
    
    # Save cleaned captions
    expanded_df[['image', 'caption', 'split']].to_csv(output_captions_file, index=False)
    logging.info(f"Cleaned captions saved to {output_captions_file}")
    
    return expanded_df
