#umich squirrels
import numpy as np

class Chunks: 
    chunk_type = tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,float]

    def __init__(self,file_name,thresh):

        self.letter_match = {
            ".": "dot", 
            "A": "cap_a",
            "B": "cap_b",
            "C": "cap_c",
            "D": "cap_d",
            "E": "cap_e",
            "F": "cap_f",
            "G": "cap_g",
            "H": "cap_h",
            "I": "cap_i",
            "J": "cap_j",
            "K": "cap_k",
            "L": "cap_l",
            "M": "cap_m",
            "N": "cap_n",
            "O": "cap_o",
            "P": "cap_p",
            "Q": "cap_q",
            "R": "cap_r",
            "S": "cap_s",
            "T": "cap_t",
            "U": "cap_u",
            "V": "cap_s",
            "W": "cap_w",
            "X": "cap_x",
            "Y": "cap_y",
            "Z": "cap_z",
            "*": "astric",
            "?":"question_mark", 
            "\"":"quotation",
            ":": "colon"
        }
        return
    
    def __json_to_dict(self, file_name):
        with open(file_name) as json_file:
            chunks_dict = json.load(json_file)
        to_ret = {}
        for key,val in chunks_dict.items():
            for i in range(len(val)) :
                val[i] = np.array(val[i])
            val = tuple(val)
            to_ret[key] = val
        return to_ret
    
    def get_chunks(self,char):
        '''
            Requires a character
            
            Return the list of chunks for a character at the origin 
        '''
        if char in self.letter_match:
            return self.chunk_dict[self.letter_match[char]]
        else:
            return self.chunk_dict[char]
        
    def simplify(self, chunks: list[chunk_type], box_height:float, thresh_degree:float) -> list[chunk_type]:
        '''
            Returns a list of chunk_type objects for one character

            Requires the original list of chunks for the character (use get_chunk), a box_height, and a threshold of similar degrees
        '''
        to_ret = []
        i = 1
        while(i < len(chunks)):
            ch_1 = chunks[i-1]
            ch_2 = chunks[i]
            if (abs(np.atan2(ch_1[4]) - np.atan2(ch_2[4])) < thresh_degree):
                slope = (ch_2[2][1] - ch_1[0][1])/(ch_2[0][0] - ch_1[2][0])
                new_chunk = (ch_1[0], ch_1[1], ch_2[2], ch_2[3], slope)
                i += 1
                to_ret.append(new_chunk)
            else:
                to_ret.append(ch_1)
            i += 1
        return to_ret
    
    def move_chunks(self, chunks:list[chunk_type], displacement:np.ndarray)->list[chunk_type]:
        '''
            Returns a list of chunks that have been moved to the location of the displacement point
            **note: if scaling and moving, scale first, then move
            Requires a list of chunks that is already at the origin 
        '''
        to_ret = []
        for c in chunks:
            moved_chunks = c + displacement
            to_ret.append(moved_chunks)
        return to_ret
    
    def scale_chunks(self, chunks: list[chunk_type], scaling_factor:float) ->list[chunk_type]:
        '''
            Returns a list of chunks which is scaled by the scaling_factor

            Requires a list of chunks at the origin 
        '''
        to_ret = []
        for c in chunks: 
            scaled_chunk = (c[0]*scaling_factor, c[1]*scaling_factor, c[2]*scaling_factor, c[3]*scaling_factor, c[4])
            to_ret.append(scaled_chunk)
        return to_ret
    
    ####TODO: upload ipad kiddos, do make_chunks.py, test scale_chunks, test move_chunks
    
    def find_chunk_match(self,point,chunks):
        '''
                
        '''
        return