with source as (

    select * from {{ ref('raw_winequality_white') }}

)

select * from source