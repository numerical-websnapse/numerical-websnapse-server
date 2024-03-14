from marshmallow import fields, Schema, ValidationError, validate, validates_schema, post_load, INCLUDE, EXCLUDE
import json, pprint, fractions

NSNP_tags = ['neurons','syn','in','out']

def string_to_float(s):
    try:
        num = float(s)
        return num
    except ValueError:
        try:
            frac = fractions.Fraction(s)
            return float(frac)
        except ValueError:
            raise ValidationError("Invalid number format")
      
class NextDataSchema(Schema):
    class Meta:
        unknown = EXCLUDE

    config = fields.List(fields.Integer(), required=True)
    spike = fields.List(fields.Integer(), required=False, allow_none=True)

    @post_load
    def convert_config_to_number(self, data, **kwargs):
        return [string_to_float(x) for x in data['config']]

class NeuronDataSchema(Schema):
    class Meta:
        unknown = EXCLUDE

    var_ = fields.List(fields.List(fields.String()), required=True)
    prf = fields.List(fields.List(fields.Raw(allow_none=True)), required=True)
    ntype = fields.String(required=True, validate=validate.OneOf(["reg", "out"]))
    label = fields.String(required=True)
    train = fields.List(fields.Raw())
    
    @validates_schema
    def validate_var_length(self, data, **kwargs):
        for var in data['var_']:
            if len(var) != 2:
                raise ValidationError("Invalid variable: missing name or value", field_name="var_")
            
    @post_load
    def convert_null_thld(self, data, **kwargs):
        for i in range(len(data['prf'])):
            if data['prf'][i][1] == '':
                data['prf'][i][1] = None
            if data['prf'][i][1] is not None:
                data['prf'][i][1] = string_to_float(data['prf'][i][1])
        return data
    
    @post_load
    def convert_var_to_number(self, data, **kwargs):
        for i in range(len(data['var_'])):
            data['var_'][i][1] = string_to_float(data['var_'][i][1])
        return data
    
    @post_load
    def convert_thld_to_number(self, data, **kwargs):
        for i in range(len(data['prf'])):
            if data['prf'][i][1] is not None:
                data['prf'][i][1] = string_to_float(data['prf'][i][1])
        return data

    @post_load
    def convert_coef_to_number(self, data, **kwargs):
        for i in range(len(data['prf'])):
            for j in range(len(data['prf'][i][2])):
                data['prf'][i][2][j][1] = string_to_float(data['prf'][i][2][j][1])
        return data

class NeuronSchema(Schema):
    class Meta:
        unknown = EXCLUDE

    id = fields.String(required=True)
    data = fields.Nested(NeuronDataSchema, required=True)
    

class SynapseSchema(Schema):
    class Meta:
        unknown = EXCLUDE

    id = fields.String(required=True)
    source = fields.String(required=True)
    target = fields.String(required=True)
    data = fields.Dict()
    

class NSNPSchema(Schema):
    class Meta:
        unknown = EXCLUDE
    
    neurons = fields.List(fields.Nested(NeuronSchema),required=True)
    syn = fields.List(fields.Nested(SynapseSchema),required=True)

class SimulationSchema(Schema):
    class Meta:
        unknown = EXCLUDE

    NSNP = fields.Nested(NSNPSchema)
    branch = fields.Int(required=True,allow_none=True)
    cur_depth = fields.Int(required=True)
    sim_depth = fields.Int(required=True)

    @validates_schema
    def validate_availability(self, data, **kwargs):
        if data['cur_depth'] > data['cur_depth']:
            raise ValidationError("Current depth is greater than simulation depth", field_name="cur_depth")

    @validates_schema
    def branch_validation(self, data, **kwargs):
        if data['branch'] is not None:
            if data['branch'] < 1:
                raise ValidationError("Invalid branch size (branch < 1)", field_name="branch")

class NSNPValidateMiddleware(object):
    def __init__(self):
        pass

    def __call__(self, req, resp, resource, params):
        req.context.err, req.context.msg = \
            self.validate(SimulationSchema(unknown=EXCLUDE),req.media)

    def validate(self,schema,data):
        try: schema.load(data);\
            return False, schema.load(data)
        except ValidationError as err:
            return True, err.messages_dict

if __name__ == "__main__":
    schema = SimulationSchema()

    with open('app/model/tests/NSNP_G6.json', 'r') as f:
        data = json.load(f)

    try:
        result = schema.load(data)
        pprint.pprint(result)
    except ValidationError as err:
        pprint.pprint(err.messages_dict)