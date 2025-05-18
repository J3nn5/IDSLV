import os
import re
import pandas as pd
import numpy as np
import logging
from io import StringIO
import joblib

class IDSPreprocessor:
    """
    Lớp tiền xử lý đầu vào cho IDS - có khả năng xử lý nhiều định dạng khác nhau 
    và chuẩn hóa đầu ra để phù hợp với các model.
    """
    
    def __init__(self, expected_features=None, scaler_path=None, log_level=logging.INFO):
        """
        Khởi tạo bộ tiền xử lý
        
        Parameters:
        -----------
        expected_features : int
            Số lượng features mà model yêu cầu
        scaler_path : str
            Đường dẫn đến file scaler đã được lưu
        log_level : int
            Mức độ logging
        """
        # Thiết lập logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("IDSPreprocessor")
        
        # Số features mà model yêu cầu
        self.expected_features = expected_features
        
        # Load scaler nếu có
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
                self.logger.info(f"Đã load scaler từ {scaler_path}")
            except Exception as e:
                self.logger.error(f"Không thể load scaler: {str(e)}")
        
        # Features quan trọng nhất (sử dụng nếu cần giảm số lượng features)
        self.important_features = None
    
    def process_file(self, filepath):
        """
        Xử lý file đầu vào và trả về dữ liệu đã được tiền xử lý
        
        Parameters:
        -----------
        filepath : str
            Đường dẫn đến file cần xử lý
            
        Returns:
        --------
        data : pandas.DataFrame
            DataFrame chứa dữ liệu đã được tiền xử lý
        metadata : dict
            Thông tin về dữ liệu và quá trình xử lý
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Không tìm thấy file {filepath}")
        
        # Thông tin về quá trình xử lý
        metadata = {
            "original_file": os.path.basename(filepath),
            "file_size": os.path.getsize(filepath),
            "original_format": os.path.splitext(filepath)[1].lower(),
            "preprocessing_steps": [],
            "warnings": [],
            "feature_count": 0,
            "row_count": 0
        }
        
        # Xác định loại file và xử lý phù hợp
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext == '.csv':
            data, meta = self._process_csv(filepath)
        elif file_ext in ['.log', '.txt']:
            data, meta = self._process_log(filepath)
        elif file_ext in ['.pcap', '.pcapng']:
            data, meta = self._process_pcap(filepath)
        elif file_ext in ['.json', '.xml']:
            data, meta = self._process_structured(filepath, file_ext[1:])
        elif file_ext in ['.xls', '.xlsx']:
            data, meta = self._process_excel(filepath)
        elif file_ext == '.parquet':
            data, meta = self._process_parquet(filepath)
        else:
            # Cố gắng đoán định dạng file dựa vào nội dung
            data, meta = self._detect_and_process(filepath)
        
        # Cập nhật metadata
        metadata.update(meta)
        
        # Chuẩn hóa dữ liệu sau khi đã xử lý theo định dạng
        data, norm_meta = self._normalize_data(data)
        metadata["preprocessing_steps"].extend(norm_meta["preprocessing_steps"])
        metadata["warnings"].extend(norm_meta["warnings"])
        
        # Điều chỉnh số lượng features nếu cần
        if self.expected_features is not None:
            data, adjust_meta = self._adjust_features(data)
            metadata["preprocessing_steps"].extend(adjust_meta["preprocessing_steps"])
            metadata["warnings"].extend(adjust_meta["warnings"])
        
        # Scale dữ liệu nếu có scaler
        if self.scaler is not None:
            try:
                # Kiểm tra xem scaler có phù hợp không
                if data.shape[1] == self.scaler.n_features_in_:
                    data_values = self.scaler.transform(data.values)
                    data = pd.DataFrame(data_values, columns=data.columns)
                    metadata["preprocessing_steps"].append("Đã scale dữ liệu")
                else:
                    msg = f"Số features ({data.shape[1]}) không khớp với scaler ({self.scaler.n_features_in_})"
                    metadata["warnings"].append(msg)
                    self.logger.warning(msg)
            except Exception as e:
                msg = f"Lỗi khi scale dữ liệu: {str(e)}"
                metadata["warnings"].append(msg)
                self.logger.error(msg)
        
        # Cập nhật thông tin cuối cùng
        metadata["feature_count"] = data.shape[1]
        metadata["row_count"] = data.shape[0]
        
        return data, metadata
    
    def _normalize_data(self, data):
        """Chuẩn hóa dữ liệu: xử lý missing values, outliers, v.v."""
        metadata = {
            "preprocessing_steps": [],
            "warnings": []
        }
        
        # Xử lý missing values
        if data.isna().any().any():
            missing_count = data.isna().sum().sum()
            metadata["preprocessing_steps"].append(f"Điền {missing_count} giá trị thiếu bằng 0")
            data = data.fillna(0)
        
        # Kiểm tra và xử lý outliers bằng phương pháp IQR
        for col in data.columns:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Đếm số lượng outliers
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            
            if outliers > 0:
                # Nếu có nhiều outliers, ghi nhận nhưng không xử lý
                if outliers > len(data) * 0.1:  # Nếu outliers > 10%
                    metadata["warnings"].append(f"Phát hiện {outliers} outliers trong cột {col}")
                # Xử lý outliers bằng cách cắt giới hạn
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                metadata["preprocessing_steps"].append(f"Xử lý outliers trong cột {col}")
        
        # Kiểm tra độ phân tán của dữ liệu
        for col in data.columns:
            if data[col].nunique() == 1:
                metadata["warnings"].append(f"Cột {col} có giá trị không đổi")
        
        return data, metadata
    
    def _process_csv(self, filepath):
        """Xử lý file CSV"""
        metadata = {
            "preprocessing_steps": ["Đọc file CSV"],
            "warnings": []
        }
        
        try:
            # Đầu tiên thử đọc với mặc định
            data = pd.read_csv(filepath)
        except Exception as e1:
            # Nếu lỗi, thử đọc với nhiều delimiter khác nhau
            self.logger.warning(f"Lỗi khi đọc CSV với mặc định: {str(e1)}")
            metadata["warnings"].append(f"Thử lại với nhiều delimiter khác nhau")
            
            for delimiter in [',', ';', '|', '\t']:
                try:
                    data = pd.read_csv(filepath, delimiter=delimiter)
                    metadata["preprocessing_steps"].append(f"Đã sử dụng delimiter: '{delimiter}'")
                    break
                except Exception:
                    continue
            else:
                # Nếu vẫn không đọc được, thử đọc từng dòng
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # Đếm số lượng ký tự phân cách trong vài dòng đầu
                    delimiters = [',', ';', '|', '\t']
                    counts = {d: sum(line.count(d) for line in lines[:10]) for d in delimiters}
                    best_delimiter = max(counts, key=counts.get)
                    
                    # Đọc lại với delimiter tốt nhất
                    data = pd.read_csv(filepath, delimiter=best_delimiter)
                    metadata["preprocessing_steps"].append(f"Đã tự động xác định delimiter: '{best_delimiter}'")
                except Exception as e2:
                    raise ValueError(f"Không thể đọc file CSV: {str(e1)}, {str(e2)}")
        
        # Xử lý header nếu không có hoặc không hợp lệ
        if data.columns.str.contains('^Unnamed').any():
            # Có thể không có header
            metadata["warnings"].append("Phát hiện cột không có tên, sử dụng header tự động")
            data.columns = [f'feature_{i}' for i in range(len(data.columns))]
        
        return self._clean_numeric_data(data, metadata)
    
    def _process_log(self, filepath):
        """Xử lý file log"""
        metadata = {
            "preprocessing_steps": ["Đọc file log"],
            "warnings": [],
            "log_format": "unknown"
        }
        
        try:
            # Đọc vài dòng đầu để phân tích định dạng
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                first_lines = [next(f) for _ in range(10) if f]
            
            # Phân tích định dạng log
            log_format = self._detect_log_format(first_lines)
            metadata["log_format"] = log_format
            
            # Chuyển đổi log thành dữ liệu có cấu trúc dựa vào định dạng
            data = self._convert_log_to_dataframe(filepath, log_format)
            metadata["preprocessing_steps"].append(f"Đã xác định định dạng log: {log_format}")
            
        except Exception as e:
            # Nếu không thể phân tích log, thử đọc như một file có cấu trúc cột
            self.logger.warning(f"Không thể phân tích log theo định dạng: {str(e)}")
            metadata["warnings"].append("Không thể phân tích log theo định dạng chuẩn, thử đọc như file văn bản có cấu trúc")
            
            try:
                # Thử đọc file như CSV với các delimiter phổ biến
                for delimiter in [' ', '\t', ',', ';']:
                    try:
                        data = pd.read_csv(filepath, delimiter=delimiter, error_bad_lines=False)
                        if data.shape[1] > 1:  # Nếu có nhiều cột, có thể đúng
                            metadata["preprocessing_steps"].append(f"Đã đọc như CSV với delimiter '{delimiter}'")
                            break
                    except Exception:
                        continue
                else:
                    # Nếu không thể, trích xuất các thông tin số từ log
                    data = self._extract_numeric_from_log(filepath)
                    metadata["preprocessing_steps"].append("Đã trích xuất thông tin số từ log")
            except Exception as e2:
                raise ValueError(f"Không thể xử lý file log: {str(e)}, {str(e2)}")
        
        return self._clean_numeric_data(data, metadata)
    
    def _process_pcap(self, filepath):
        """Xử lý file PCAP (cần thư viện scapy)"""
        metadata = {
            "preprocessing_steps": ["Đọc file PCAP"],
            "warnings": []
        }
        
        try:
            # Kiểm tra xem có scapy không
            try:
                import scapy.all as scapy
                from scapy.utils import RawPcapReader
            except ImportError:
                metadata["warnings"].append("Thư viện scapy không có sẵn, sử dụng tshark để xử lý PCAP")
                # Sử dụng tshark làm phương pháp dự phòng
                return self._process_pcap_with_tshark(filepath)
            
            # Xử lý với scapy
            packets = []
            pcap_reader = RawPcapReader(filepath)
            
            for packet_data, metadata_pcap in pcap_reader:
                try:
                    packet = scapy.Ether(packet_data)
                    packet_dict = {}
                    
                    # Trích xuất metadata của packet
                    packet_dict["time"] = metadata_pcap.sec + metadata_pcap.usec / 1000000
                    packet_dict["len"] = len(packet)
                    
                    # Trích xuất header IP nếu có
                    if scapy.IP in packet:
                        ip = packet[scapy.IP]
                        packet_dict["ip_src"] = int(ip.src.split('.')[-1])
                        packet_dict["ip_dst"] = int(ip.dst.split('.')[-1])
                        packet_dict["ip_ttl"] = ip.ttl
                        packet_dict["ip_proto"] = ip.proto
                    
                    # Trích xuất thông tin TCP/UDP nếu có
                    if scapy.TCP in packet:
                        tcp = packet[scapy.TCP]
                        packet_dict["src_port"] = tcp.sport
                        packet_dict["dst_port"] = tcp.dport
                        packet_dict["tcp_flags"] = int(tcp.flags)
                    elif scapy.UDP in packet:
                        udp = packet[scapy.UDP]
                        packet_dict["src_port"] = udp.sport
                        packet_dict["dst_port"] = udp.dport
                    
                    packets.append(packet_dict)
                except Exception as e:
                    self.logger.debug(f"Bỏ qua packet: {str(e)}")
            
            # Tạo DataFrame từ packets
            data = pd.DataFrame(packets)
            metadata["preprocessing_steps"].append(f"Đã xử lý {len(packets)} packets")
            
        except Exception as e:
            # Thử sử dụng tshark nếu scapy lỗi
            self.logger.warning(f"Lỗi khi xử lý PCAP với scapy: {str(e)}")
            metadata["warnings"].append(f"Thử lại với tshark: {str(e)}")
            return self._process_pcap_with_tshark(filepath)
        
        return self._clean_numeric_data(data, metadata)
    
    def _process_pcap_with_tshark(self, filepath):
        """Xử lý PCAP bằng tshark (Wireshark CLI)"""
        metadata = {
            "preprocessing_steps": ["Đọc file PCAP với tshark"],
            "warnings": []
        }
        
        try:
            import subprocess
            
            # Thực thi tshark để xuất dữ liệu dạng CSV
            tshark_output = subprocess.check_output([
                'tshark', '-r', filepath, '-T', 'fields',
                '-e', 'frame.time_epoch', '-e', 'ip.src', '-e', 'ip.dst',
                '-e', 'tcp.srcport', '-e', 'tcp.dstport', '-e', 'udp.srcport',
                '-e', 'udp.dstport', '-e', 'ip.proto', '-e', 'frame.len',
                '-E', 'header=y', '-E', 'separator=,'
            ], stderr=subprocess.PIPE, universal_newlines=True)
            
            # Chuyển đổi output thành DataFrame
            data = pd.read_csv(StringIO(tshark_output))
            metadata["preprocessing_steps"].append(f"Đã xử lý {len(data)} packets qua tshark")
            
        except Exception as e:
            self.logger.error(f"Không thể xử lý PCAP với tshark: {str(e)}")
            metadata["warnings"].append("Không thể xử lý file PCAP, tạo features cơ bản")
            
            # Tạo một tập dữ liệu đơn giản từ metadata của file
            file_stats = os.stat(filepath)
            
            # Tạo một số features đơn giản từ metadata
            data = pd.DataFrame({
                'file_size': [file_stats.st_size],
                'last_modified': [file_stats.st_mtime]
            })
            
            for i in range(20):
                # Thêm một số features ngẫu nhiên để đủ số lượng cho model
                data[f'feature_{i}'] = np.random.randn(1)
                
            metadata["warnings"].append("Sử dụng dữ liệu giả khi không thể phân tích PCAP")
        
        return self._clean_numeric_data(data, metadata)
    
    def _process_structured(self, filepath, format_type):
        """Xử lý file có cấu trúc như JSON, XML"""
        metadata = {
            "preprocessing_steps": [f"Đọc file {format_type.upper()}"],
            "warnings": []
        }
        
        try:
            if format_type == 'json':
                data = pd.read_json(filepath)
            elif format_type == 'xml':
                # Yêu cầu thư viện lxml
                try:
                    import lxml
                    import xml.etree.ElementTree as ET
                    
                    # Đọc XML và chuyển đổi thành dict
                    tree = ET.parse(filepath)
                    root = tree.getroot()
                    
                    # Chuyển đổi XML thành list of dicts
                    records = []
                    for element in root:
                        record = {}
                        for child in element:
                            record[child.tag] = child.text
                        records.append(record)
                    
                    data = pd.DataFrame(records)
                except ImportError:
                    metadata["warnings"].append("Thư viện lxml không có sẵn")
                    # Đọc như text và trích xuất thông tin
                    with open(filepath, 'r', encoding='utf-8') as f:
                        xml_content = f.read()
                    
                    # Trích xuất các giá trị từ XML bằng regex
                    values = re.findall(r'<[^>]+>([^<]+)</[^>]+>', xml_content)
                    
                    # Tạo DataFrame với features là giá trị
                    data = pd.DataFrame({f'value_{i}': [v] for i, v in enumerate(values)})
            else:
                raise ValueError(f"Không hỗ trợ định dạng {format_type}")
                
        except Exception as e:
            # Nếu lỗi, đọc như text
            self.logger.warning(f"Lỗi khi đọc {format_type.upper()}: {str(e)}")
            metadata["warnings"].append(f"Không thể đọc như {format_type.upper()}, thử đọc như text")
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Trích xuất tất cả các số từ nội dung
                numbers = re.findall(r'\d+\.\d+|\d+', content)
                
                # Tạo DataFrame từ các số
                data = pd.DataFrame({f'value_{i}': [float(n)] for i, n in enumerate(numbers[:100])})
                
                metadata["preprocessing_steps"].append(f"Đã trích xuất {len(numbers)} giá trị số từ {format_type}")
            except Exception as e2:
                raise ValueError(f"Không thể xử lý file {format_type}: {str(e)}, {str(e2)}")
        
        return self._clean_numeric_data(data, metadata)
    
    def _process_excel(self, filepath):
        """Xử lý file Excel"""
        metadata = {
            "preprocessing_steps": ["Đọc file Excel"],
            "warnings": []
        }
        
        try:
            # Đọc sheet đầu tiên mặc định
            data = pd.read_excel(filepath)
            metadata["preprocessing_steps"].append("Đã đọc sheet đầu tiên")
            
            # Kiểm tra nếu có nhiều sheet
            excel = pd.ExcelFile(filepath)
            if len(excel.sheet_names) > 1:
                metadata["warnings"].append(f"File có {len(excel.sheet_names)} sheets, chỉ đọc sheet đầu tiên")
                metadata["all_sheets"] = excel.sheet_names
        except Exception as e:
            self.logger.error(f"Không thể đọc file Excel: {str(e)}")
            raise ValueError(f"Không thể đọc file Excel: {str(e)}")
        
        return self._clean_numeric_data(data, metadata)
    
    def _process_parquet(self, filepath):
        """Xử lý file Parquet"""
        metadata = {
            "preprocessing_steps": ["Đọc file Parquet"],
            "warnings": []
        }
        
        try:
            # Sử dụng thư viện pyarrow hoặc fastparquet
            import pyarrow.parquet as pq
            
            # Đọc file Parquet
            table = pq.read_table(filepath)
            data = table.to_pandas()
            metadata["preprocessing_steps"].append(f"Đã đọc {len(data)} bản ghi từ file Parquet")
            
            # Kiểm tra và xử lý các cột phức tạp (dictionary, list, etc.)
            complex_cols = []
            for col in data.columns:
                if data[col].dtype == 'object':
                    try:
                        # Thử chuyển đổi sang số nếu có thể
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    except:
                        complex_cols.append(col)
            
            if complex_cols:
                metadata["warnings"].append(f"Phát hiện {len(complex_cols)} cột phức tạp, sẽ loại bỏ")
                data = data.drop(columns=complex_cols)
            
            # Đảm bảo đúng định dạng số
            for col in data.select_dtypes(include=['bool']).columns:
                data[col] = data[col].astype(int)
                
        except ImportError:
            metadata["warnings"].append("Thư viện pyarrow không có sẵn, sử dụng pandas")
            try:
                data = pd.read_parquet(filepath)
                metadata["preprocessing_steps"].append(f"Đã đọc {len(data)} bản ghi từ file Parquet với pandas")
            except Exception as e:
                raise ValueError(f"Không thể đọc file Parquet: {str(e)}")
        
        return self._clean_numeric_data(data, metadata)
    
    def _detect_and_process(self, filepath):
        """Phát hiện định dạng file dựa vào nội dung và xử lý phù hợp"""
        metadata = {
            "preprocessing_steps": ["Tự động phát hiện định dạng file"],
            "warnings": [],
            "detected_format": "unknown"
        }
        
        try:
            # Đọc vài dòng đầu để phân tích
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                first_chunk = f.read(4096)  # Đọc 4KB đầu tiên
            
            # Kiểm tra xem có phải CSV không
            if ',' in first_chunk and '\n' in first_chunk:
                comma_count = first_chunk.count(',')
                newline_count = first_chunk.count('\n')
                
                if comma_count > newline_count:  # Nhiều dấu phẩy hơn dòng mới
                    metadata["detected_format"] = "csv"
                    return self._process_csv(filepath)
            
            # Kiểm tra xem có phải JSON không
            if first_chunk.strip().startswith('{') and '}' in first_chunk:
                metadata["detected_format"] = "json"
                return self._process_structured(filepath, 'json')
            
            # Kiểm tra xem có phải XML không
            if first_chunk.strip().startswith('<') and '>' in first_chunk:
                metadata["detected_format"] = "xml"
                return self._process_structured(filepath, 'xml')
            
            # Nếu không xác định được, đọc như log
            metadata["detected_format"] = "log"
            metadata["warnings"].append("Không thể xác định định dạng file, xử lý như log")
            return self._process_log(filepath)
            
        except Exception as e:
            self.logger.error(f"Lỗi khi tự động phát hiện định dạng: {str(e)}")
            metadata["warnings"].append(f"Không thể tự động phát hiện định dạng: {str(e)}")
            
            # Cố gắng trích xuất dữ liệu số từ file
            try:
                with open(filepath, 'rb') as f:
                    content = f.read().decode('utf-8', errors='ignore')
                
                # Trích xuất tất cả số từ nội dung
                numbers = re.findall(r'\d+\.\d+|\d+', content)
                
                # Tạo DataFrame từ các số
                data = pd.DataFrame({f'feature_{i}': [float(n)] for i, n in enumerate(numbers[:100])})
                metadata["preprocessing_steps"].append(f"Đã trích xuất {len(numbers)} giá trị số")
                
                return data, metadata
            except Exception as e2:
                raise ValueError(f"Không thể xử lý file: {str(e)}, {str(e2)}")
    
    def _detect_log_format(self, lines):
        """Phát hiện định dạng log từ các dòng mẫu"""
        # Thử khớp với các định dạng log phổ biến
        formats = {
            'syslog': r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}',
            'apache': r'\d+\.\d+\.\d+\.\d+.*\[.*\].*HTTP',
            'nginx': r'\d+\.\d+\.\d+\.\d+.*\[.*\].*"(GET|POST|PUT|DELETE)',
            'ids': r'(\d+\.\d+\.\d+\.\d+|\[\d+\.\d+\.\d+\.\d+\]).*alert'
        }
        
        # Đếm số dòng khớp với mỗi định dạng
        format_counts = {fmt: sum(1 for line in lines if re.search(pattern, line)) for fmt, pattern in formats.items()}
        
        # Trả về định dạng phổ biến nhất
        if any(format_counts.values()):
            return max(format_counts, key=format_counts.get)
        return "unknown"
    
    def _convert_log_to_dataframe(self, filepath, log_format):
        """Chuyển đổi file log thành DataFrame dựa vào định dạng"""
        if log_format == "syslog":
            # Ví dụ mẫu: "May 2 12:34:56 host process[123]: message"
            pattern = r'(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+([^:]+)\[?(\d*)\]?:\s+(.*)'
            columns = ['timestamp', 'host', 'process', 'pid', 'message']
        elif log_format == "apache":
            # Ví dụ mẫu: "192.168.1.1 - user [01/Jan/2021:12:34:56 +0000] "GET /page HTTP/1.1" 200 1234"
            pattern = r'(\S+) \S+ \S+ \[(.*?)\] "(.*?)" (\d+) (\d+)'
            columns = ['ip', 'timestamp', 'request', 'status', 'bytes']
        elif log_format == "nginx":
            # Tương tự Apache
            pattern = r'(\S+) \S+ \S+ \[(.*?)\] "(.*?)" (\d+) (\d+)'
            columns = ['ip', 'timestamp', 'request', 'status', 'bytes']
        elif log_format == "ids":
            # Pattern cho log IDS/IPS phổ biến
            pattern = r'.*alert.*\[\**(\d+):(\d+):(\d+)\].*\{(.*?)\}.*'
            columns = ['sig_id', 'sig_rev', 'sig_class', 'protocol']
        else:
            # Định dạng không xác định, trích xuất các thông tin số
            return self._extract_numeric_from_log(filepath)
        
        # Đọc file và trích xuất thông tin
        records = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                match = re.search(pattern, line)
                if match:
                    records.append(list(match.groups()))
        
        # Tạo DataFrame
        df = pd.DataFrame(records, columns=columns)
        
        # Chuyển đổi các cột số
        for col in df.columns:
            if col in ['status', 'bytes', 'pid', 'sig_id', 'sig_rev']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Tách IP thành các phần
        if 'ip' in df.columns:
            try:
                ip_parts = df['ip'].str.split('.', expand=True)
                for i in range(min(4, ip_parts.shape[1])):
                    df[f'ip_part_{i+1}'] = pd.to_numeric(ip_parts[i], errors='coerce')
            except Exception:
                pass
        
        # Xử lý timestamp
        if 'timestamp' in df.columns:
            try:
                # Chuyển đổi sang UNIX timestamp nếu có thể
                if log_format in ["apache", "nginx"]:
                    # Format: 01/Jan/2021:12:34:56
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%b/%Y:%H:%M:%S', errors='coerce')
                elif log_format == "syslog":
                    # Format: May 2 12:34:56
                    current_year = pd.Timestamp.now().year
                    df['timestamp'] = pd.to_datetime(f"{current_year} " + df['timestamp'], format='%Y %b %d %H:%M:%S', errors='coerce')
                
                # Tạo features từ timestamp
                df['hour'] = df['timestamp'].dt.hour
                df['minute'] = df['timestamp'].dt.minute
                df['second'] = df['timestamp'].dt.second
                df['weekday'] = df['timestamp'].dt.weekday
                
                # Chuyển sang UNIX timestamp
                df['timestamp'] = df['timestamp'].astype('int64') // 10**9
            except Exception:
                pass
        
        # Xử lý cột request nếu có
        if 'request' in df.columns:
            try:
                request_parts = df['request'].str.split(' ', n=2, expand=True)
                if request_parts.shape[1] >= 3:
                    df['method'] = request_parts[0]
                    df['path'] = request_parts[1]
                    df['http_version'] = request_parts[2]
                    
                    # One-hot encoding cho method
                    methods = ['GET', 'POST', 'PUT', 'DELETE']
                    for method in methods:
                        df[f'method_{method}'] = (df['method'] == method).astype(int)
            except Exception:
                pass
        
        return df
    
    def _extract_numeric_from_log(self, filepath):
        """Trích xuất các thông tin số từ file log không có cấu trúc rõ ràng"""
        # Patterns để tìm các thông tin hữu ích
        patterns = {
            'ip': r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b',
            'timestamp': r'\b(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})\b',
            'number': r'\b(\d+)\b',
            'decimal': r'\b(\d+\.\d+)\b',
            'port': r':(\d{1,5})\b',
            'http_status': r'\b([1-5][0-9][0-9])\b'
        }
        
        records = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f):
                record = {'line_num': line_num}
                
                # Trích xuất các thông tin theo pattern
                for name, pattern in patterns.items():
                    matches = re.findall(pattern, line)
                    if name == 'ip' and matches:
                        # Tách IP và chuyển thành số
                        ip = matches[0]
                        parts = ip.split('.')
                        for i, part in enumerate(parts):
                            record[f'ip_part_{i+1}'] = int(part)
                    elif name in ['number', 'decimal', 'port', 'http_status'] and matches:
                        # Lưu các giá trị số
                        for i, val in enumerate(matches[:5]):  # Giới hạn 5 giá trị
                            record[f'{name}_{i+1}'] = float(val) if '.' in val else int(val)
                
                # Thêm độ dài dòng và số kí tự đặc biệt
                record['line_length'] = len(line)
                record['special_chars'] = sum(c in '{}[]()<>:;"\'' for c in line)
                
                records.append(record)
        
        # Tạo DataFrame
        df = pd.DataFrame(records)
        
        # Đảm bảo df có ít nhất 1 dòng
        if len(df) == 0:
            df = pd.DataFrame({'line_num': [0], 'line_length': [0], 'special_chars': [0]})
            for i in range(20):
                df[f'feature_{i}'] = np.random.randn(1)
        
        return df, {"preprocessing_steps": [f"Đã trích xuất thông tin số từ {len(records)} dòng log"], "warnings": []}
    
    def _clean_numeric_data(self, data, metadata):
        """Làm sạch dữ liệu, chỉ giữ lại các cột số"""
        original_shape = data.shape
        
        # Chỉ giữ lại các cột số
        numeric_data = data.select_dtypes(include=[np.number])
        
        # Thêm thông tin về việc loại bỏ các cột không phải số
        if numeric_data.shape[1] < original_shape[1]:
            dropped_cols = original_shape[1] - numeric_data.shape[1]
            metadata["preprocessing_steps"].append(f"Đã loại bỏ {dropped_cols} cột không phải số")
        
        # Thay thế giá trị NaN bằng 0
        if numeric_data.isna().any().any():
            missing_values = numeric_data.isna().sum().sum()
            metadata["preprocessing_steps"].append(f"Đã thay thế {missing_values} giá trị thiếu bằng 0")
            numeric_data = numeric_data.fillna(0)
        
        # Loại bỏ các cột hằng số (không có giá trị phân biệt)
        constant_cols = [col for col in numeric_data.columns if numeric_data[col].nunique() <= 1]
        if constant_cols:
            metadata["preprocessing_steps"].append(f"Đã loại bỏ {len(constant_cols)} cột hằng số")
            numeric_data = numeric_data.drop(columns=constant_cols)
        
        # Nếu không còn cột nào, thêm cột giả
        if numeric_data.shape[1] == 0:
            metadata["warnings"].append("Không có cột số nào, tạo cột giả")
            for i in range(20):
                numeric_data[f'feature_{i}'] = np.random.randn(len(data))
        
        return numeric_data, metadata
    
    def _adjust_features(self, data):
        """Điều chỉnh số lượng features cho phù hợp với yêu cầu của model"""
        metadata = {
            "preprocessing_steps": [],
            "warnings": []
        }
        
        current_features = data.shape[1]
        
        if current_features == self.expected_features:
            return data, metadata
        
        if current_features < self.expected_features:
            # Thêm các cột 0 nếu thiếu features
            missing_features = self.expected_features - current_features
            metadata["preprocessing_steps"].append(f"Thêm {missing_features} cột bổ sung")
            
            for i in range(missing_features):
                data[f'add_feature_{i}'] = 0
                
        elif current_features > self.expected_features:
            # Có quá nhiều features, cần giảm bớt
            excess_features = current_features - self.expected_features
            metadata["preprocessing_steps"].append(f"Giảm {excess_features} cột thừa")
            
            if self.important_features is not None:
                # Nếu có danh sách features quan trọng, sử dụng nó
                selected_features = [col for col in self.important_features if col in data.columns][:self.expected_features]
                if len(selected_features) < self.expected_features:
                    # Nếu không đủ features, bổ sung thêm
                    remaining = self.expected_features - len(selected_features)
                    other_features = [col for col in data.columns if col not in selected_features][:remaining]
                    selected_features.extend(other_features)
            else:
                # Nếu không có danh sách features quan trọng, chọn ngẫu nhiên
                selected_features = list(data.columns[:self.expected_features])
                
            data = data[selected_features]
            
        return data, metadata

def create_input_processor():
    """Tạo một bộ tiền xử lý IDS với cấu hình đã định sẵn"""
    return IDSPreprocessor(
        expected_features=196,  # Số features mà model yêu cầu
        scaler_path="models/scaler.pkl",
        log_level=logging.INFO
    )