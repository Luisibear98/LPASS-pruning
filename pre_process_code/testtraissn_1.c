 int dns_read_name(unsigned char *buffer, unsigned char *bufend,
                  unsigned char *name, char *destination, int dest_len,
                 int *offset)
 {
        int nb_bytes = 0, n = 0;
        int label_len;
	unsigned char *reader = name;
	char *dest = destination;
	while (1) {
		if ((*reader & 0xc0) == 0xc0) {
                        if ((buffer + reader[1]) > reader)
                                goto err;
                        n = dns_read_name(buffer, bufend, buffer + reader[1],
                                         dest, dest_len - nb_bytes, offset);
                        if (n == 0)
                                goto err;
		}
		label_len = *reader;
		if (label_len == 0)
			goto out;
		if ((reader + label_len >= bufend) || (nb_bytes + label_len >= dest_len))
			goto err;
		label_len++;
		memcpy(dest, reader, label_len);
		dest     += label_len;
		nb_bytes += label_len;
		reader   += label_len;
	}
  out:
	reader  = name;
	*offset = 0;
	while (reader < bufend) {
		if ((reader[0] & 0xc0) == 0xc0) {
			*offset += 2;
			break;
		}
		else if (*reader == 0) {
			*offset += 1;
			break;
		}
		*offset += 1;
		++reader;
	}
	return nb_bytes;
  err:
	return 0;
}
